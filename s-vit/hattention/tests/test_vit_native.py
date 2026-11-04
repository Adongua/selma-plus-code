"""Small three-party check of the native full-forward baseline."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

from hattention.benchmark_vit_native_cuda import (AuditedPlainOperators, CrypTenNativeOperators,
                                                  PlainStandardOperators, chebyshev_gelu, numerical_gate,
                                                  scaled_standard_attention, validate_reciprocal_scale)
from hattention.diagnose_vit_native_attention import open_exact, trace_attention
from hattention.exact_decode import reveal_exact
from hattention.experimental_crypten_truncation import experimental_truncation
from hattention.offline_materials import HostCacheBudget, StrictProviderCache
from cryptgpu_gelu.operators import _chebyshev_coefficients, chebyshev16_gelu_reference
from hattention.config import HAttentionConfig
from hattention.student_checkpoint import import_mpcvit
from hattention.vit_executor import forward_vit
from hattention.vit_integration import make_fixed_student

ROOT = Path(__file__).resolve().parents[3]


def _worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = 20
    crypten.init(device=torch.device("cpu"))
    try:
        import_mpcvit(ROOT / "mpcvit")
        from src.vit import ViTLite
        torch.manual_seed(25)
        source = ViTLite(img_size=8, kernel_size=4, embedding_dim=12, num_layers=1,
                         num_heads=3, mlp_ratio=2, num_classes=3, dropout=0,
                         attention_dropout=0, stochastic_depth=0)
        config = HAttentionConfig(3, ("rsa", "lsa", "mixer"), patch_grid=(2, 2))
        model = make_fixed_student(source, [config]).eval()
        image = torch.randn(1, 3, 8, 8) * 0.1
        reference = forward_vit(model, image, PlainStandardOperators())
        encrypted = crypten.cryptensor(image, src=0, precision=20)
        actual = forward_vit(model, encrypted, CrypTenNativeOperators()).get_plain_text()
        if not torch.isfinite(actual).all() or (actual - reference).abs().max() > 1.0:
            raise AssertionError(f"native ViT mismatch: {actual} versus {reference}")
        coefficients = _chebyshev_coefficients(16, 12.0)
        fixture = torch.tensor([-12., -8.426, -1., 0., 1., 8.426, 12.])
        for precision in (17, 20):
            crypten.cfg.encoder.precision_bits = precision
            secret = crypten.cryptensor(fixture, src=0, precision=precision)
            observed = chebyshev_gelu(secret, 12.0, coefficients).get_plain_text()
            expected = chebyshev_gelu(fixture, 12.0, coefficients)
            if not torch.isfinite(observed).all() or (observed - expected).abs().max() > 0.05:
                raise AssertionError(f"Q{precision} Chebyshev mismatch: {observed} versus {expected}")
        crypten.cfg.encoder.precision_bits = 20
        plaintext_ops = AuditedPlainOperators(12.0)
        sites = {}
        approximate = forward_vit(model, image, plaintext_ops,
                                   on_stage=lambda i, s, x: sites.update({(i, s): x.detach().clone()}))
        encrypted = crypten.cryptensor(image, src=0, precision=20)
        opened_sites = []
        def inspect(i, site, value):
            opened = value.get_plain_text()
            if opened.shape != sites[(i, site)].shape or not torch.isfinite(opened).all():
                raise AssertionError(f"invalid opened site {i} {site}")
            opened_sites.append((i, site))
        cheb_output = forward_vit(model, encrypted, CrypTenNativeOperators("chebyshev16", 12.0),
                                  on_stage=inspect).get_plain_text()
        if opened_sites != list(sites) or len(opened_sites) != 7:
            raise AssertionError(f"unexpected opened site order: {opened_sites}")
        if not torch.isfinite(cheb_output).all() or (cheb_output - approximate).abs().max() > 1.0:
            raise AssertionError(f"Chebyshev ViT mismatch: {cheb_output} versus {approximate}")
        variance = torch.tensor([0.00689607485])
        expected_sqrt = variance.sqrt()
        roots = []
        for iterations in (3, 7):
            crypten.cfg.functions.sqrt_nr_iters = iterations
            encrypted_variance = crypten.cryptensor(variance, src=0, precision=20)
            roots.append(encrypted_variance.sqrt().get_plain_text())
        if not ((roots[1] - expected_sqrt).abs().max() < 0.001 and
                (roots[1] - expected_sqrt).abs().max() < (roots[0] - expected_sqrt).abs().max()):
            raise AssertionError(f"unexpected Newton convergence: {roots} vs {expected_sqrt}")
        q, k, v = [torch.randn(1, 2, 5, 4) * 0.1 for _ in range(3)]
        secret_q, secret_k, secret_v = [crypten.cryptensor(t, src=0, precision=20) for t in (q, k, v)]
        rows = []
        extra = trace_attention(crypten, secret_q, secret_k, secret_v,
                                q.double(), k.double(), v.double(), rows.append)
        if (len(rows) != 11 or any(not row["finite"] for row in rows) or
                extra["denominator_nonpositive"] or extra["qkv_score_reconstruction_max_abs"] > 0.01):
            raise AssertionError(f"unexpected softmax trace: {rows}, {extra}")
        traced = []
        trace_attention(crypten, secret_q, secret_k, secret_v, q.double(), k.double(),
                        v.double(), traced.append, trace_exp=True)
        if (len(traced) != 14 + crypten.cfg.functions.exp_iterations or
                not any(row["stage"] == "exp_square_1" for row in traced) or
                any(not row["finite"] for row in traced)):
            raise AssertionError(f"unexpected exponential trace: {traced}")
        trace_map = {row["stage"]: row for row in traced}
        for stage in ("exp_division", "exp_public_addition"):
            if trace_map[stage]["max_abs"] > 0.0001 or len(trace_map[stage]["max_error_index"]) != 4:
                raise AssertionError(f"unexpected {stage} diagnostic: {trace_map[stage]}")
        encoded_boundary = torch.tensor([-2.0 + 2**-20], dtype=torch.float64)
        secret_boundary = crypten.cryptensor(encoded_boundary, src=0, precision=20)
        _, decoded_exact, decoded_legacy = open_exact(secret_boundary)
        decoded_forward = reveal_exact(secret_boundary)
        if not torch.equal(decoded_forward, decoded_exact):
            raise AssertionError("forward reconstruction must decode raw fixed-point integers exactly")
        if not torch.allclose(decoded_exact, encoded_boundary, atol=0, rtol=0) or not torch.allclose(
                decoded_legacy - decoded_exact, -torch.ones_like(decoded_exact), atol=1e-6, rtol=0):
            raise AssertionError(f"negative-boundary decoding not detected: {decoded_exact}, {decoded_legacy}")
        raw_traced = []
        trace_attention(crypten, secret_q, secret_k, secret_v, q.double(), k.double(),
                        v.double(), raw_traced.append, trace_exp=True, trace_raw_square=True)
        probes = [row["raw_square_probe"] for row in raw_traced if "raw_square_probe" in row]
        if (len(probes) != crypten.cfg.functions.exp_iterations or
                any(p["raw_product_mismatches"] or p["truncation_deltas_over_4"] for p in probes)):
            raise AssertionError(f"unexpected raw-square diagnostics: {probes}")
        qk_rows = []
        trace_attention(crypten, secret_q, secret_k, secret_v, q.double(), k.double(),
                        v.double(), qk_rows.append, trace_raw_qk=True)
        qk_probes = [row["raw_qk_probe"] for row in qk_rows if "raw_qk_probe" in row]
        if (len(qk_probes) != 2 or any(p["raw_product_mismatches"] or
                p["truncation_deltas_over_4"] for p in qk_probes)):
            raise AssertionError(f"unexpected raw QK diagnostics: {qk_probes}")
        scaled_rows = []
        scaled = trace_attention(crypten, secret_q, secret_k, secret_v, q.double(), k.double(),
                                 v.double(), scaled_rows.append, reciprocal_scale=256)
        if (len(scaled_rows) != 15 or any(not row["finite"] for row in scaled_rows) or
                scaled["reciprocal_scale"] != 256 or scaled["reciprocal_residual_max_abs"] > 0.03):
            raise AssertionError(f"unexpected scaled reciprocal: {scaled_rows}, {scaled}")
        output = scaled_standard_attention(secret_q, secret_k, secret_v, 256).get_plain_text()
        target = torch.softmax(q.matmul(k.transpose(-2, -1)) * 0.5, dim=-1).matmul(v)
        if not torch.isfinite(output).all() or (output - target).abs().max() > 0.03:
            raise AssertionError(f"scaled attention mismatch: {output} vs {target}")
        encrypted = crypten.cryptensor(image, src=0, precision=20)
        output = forward_vit(model, encrypted,
                             CrypTenNativeOperators("chebyshev16", 12.0, "scaled", 256)).get_plain_text()
        if not torch.isfinite(output).all() or (output - approximate).abs().max() > 1.0:
            raise AssertionError(f"scaled full forward mismatch: {output} vs {approximate}")
        for mode in ("experimental", "batched", "carry_save"):
            encrypted = crypten.cryptensor(image, src=0, precision=20)
            with experimental_truncation(mode):
                output = reveal_exact(forward_vit(model, encrypted,
                                      CrypTenNativeOperators("chebyshev16", 12.0, "scaled", 256)))
            if not torch.isfinite(output).all() or (output - approximate).abs().max() > 1.0:
                raise AssertionError(f"{mode} full forward mismatch: {output} vs {approximate}")
        # Trace the public-shape call sequence, then execute two fresh,
        # strictly precomputed full forwards. A miss or reuse must fail closed.
        budget = HostCacheBudget(1 << 28)
        cache = StrictProviderCache(crypten, "cpu", torch.device("cpu"), budget)
        cache.install()
        try:
            encrypted = crypten.cryptensor(image, src=0, precision=20)
            cache.begin_trace()
            with experimental_truncation("carry_save"):
                traced_output = reveal_exact(forward_vit(model, encrypted,
                    CrypTenNativeOperators("chebyshev16", 12.0, "scaled", 256)))
            cache.end_trace()
            if not cache.requests or not torch.isfinite(traced_output).all():
                raise AssertionError("provider trace did not capture a valid forward")
            for _ in range(2):
                cache.prepare()
                cache.begin_replay()
                with experimental_truncation("carry_save"):
                    output = reveal_exact(forward_vit(model, encrypted,
                        CrypTenNativeOperators("chebyshev16", 12.0, "scaled", 256)))
                cache.end_replay()
                if budget.used_bytes or budget.peak_bytes <= 0:
                    raise AssertionError("provider host cache accounting mismatch")
                if not torch.isfinite(output).all() or (output - approximate).abs().max() > 1.0:
                    raise AssertionError("precomputed provider full forward mismatch")
            try:
                cache.begin_replay()
            except RuntimeError:
                pass
            else:
                raise AssertionError("consumed materials were accepted a second time")
        finally:
            cache.close()
    finally:
        crypten.uninit()


class NativeViTTests(unittest.TestCase):
    def test_reciprocal_scale_validation(self):
        self.assertEqual(validate_reciprocal_scale(256), 256)
        for invalid in (0, -1, 3, True, 2**21):
            with self.assertRaises(ValueError):
                validate_reciprocal_scale(invalid)

    def test_gate_uses_selected_formula_and_excludes_warmup(self):
        rows = [{"warmup": True, "error_vs_own_formula_logits": {"finite": True, "max_abs": 100.}},
                {"warmup": False, "error_vs_plaintext_logits": {"finite": True, "max_abs": 1.1},
                 "error_vs_own_formula_logits": {"finite": True, "max_abs": 0.1}}]
        self.assertTrue(numerical_gate(rows, 1.0))
        self.assertFalse(numerical_gate(rows, 1.0, include_warmup=True))
        self.assertFalse(numerical_gate(rows, 0.01))
        self.assertFalse(numerical_gate(rows[:1], 1.0))

    def test_chebyshev_reference_and_fit_domain(self):
        for bound, tolerance in ((8.0, 0.018), (12.0, 0.083)):
            values = torch.linspace(-bound, bound, 10001, dtype=torch.float64)
            result = chebyshev_gelu(values, bound, _chebyshev_coefficients(16, bound))
            self.assertLess((result - torch.nn.functional.gelu(values)).abs().max(), tolerance)
            if bound == 8.0:
                torch.testing.assert_close(result, chebyshev16_gelu_reference(values))

    def test_three_party_native_full_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(Path(directory, "store").as_uri(),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
