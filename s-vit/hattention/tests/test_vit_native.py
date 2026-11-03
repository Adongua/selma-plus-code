"""Small three-party check of the native full-forward baseline."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

from hattention.benchmark_vit_native_cuda import (AuditedPlainOperators, CrypTenNativeOperators,
                                                  PlainStandardOperators, chebyshev_gelu, numerical_gate)
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
    finally:
        crypten.uninit()


class NativeViTTests(unittest.TestCase):
    def test_gate_uses_selected_formula_and_excludes_warmup(self):
        rows = [{"warmup": True, "error_vs_own_formula_logits": {"finite": True, "max_abs": 100.}},
                {"warmup": False, "error_vs_plaintext_logits": {"finite": True, "max_abs": 1.1},
                 "error_vs_own_formula_logits": {"finite": True, "max_abs": 0.1}}]
        self.assertTrue(numerical_gate(rows, 1.0))
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
