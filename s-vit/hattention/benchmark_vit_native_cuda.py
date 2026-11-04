"""Three-party full ViTLite forward with CrypTen-native nonlinear operators.

This uses MPCViT checkpoint weights with every attention head replaced by
standard softmax. It is an operator-cost control, not a trained standard ViT.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import torch
import torch.multiprocessing as mp

from sgelu.baseline import crypten_erf_gelu
from cryptgpu_gelu.operators import _chebyshev_coefficients
from slayernorm.baselines import crypten_native_layernorm
from slayernorm.benchmark_crypten_cuda import _crypten_measure, _plain
from .benchmark_compare_cuda import standard_attention
from .benchmark_vit_cuda import SecureOperators, image_tensor, load_model
from .student_checkpoint import file_sha256
from .vit_executor import PlainOperators, forward_vit
from .exact_decode import reveal_exact
from .experimental_crypten_truncation import experimental_truncation
from .offline_materials import StrictProviderCache


class PlainStandardOperators(PlainOperators):
    def attention(self, q, k, v, module):
        return torch.softmax(q.matmul(k.transpose(-2, -1)) * (q.size(-1) ** -0.5), dim=-1).matmul(v)


def chebyshev_gelu(value, bound, coefficients):
    """Clenshaw evaluation on torch.Tensor or CrypTen MPCTensor."""
    z = value.mul(1.0 / bound)
    previous = value.mul(0)
    previous2 = previous.clone()
    for coefficient in reversed(coefficients[1:]):
        current = z.mul(previous).mul(2).sub(previous2).add(coefficient)
        previous2, previous = previous, current
    return z.mul(previous).sub(previous2).add(0.5 * coefficients[0])


def numerical_gate(records, max_abs_error, *, include_warmup=False):
    measured = [r for r in records if include_warmup or not r["warmup"]]
    return bool(measured) and all(r["error_vs_own_formula_logits"]["finite"] and
                                  r["error_vs_own_formula_logits"]["max_abs"] <= max_abs_error
                                  for r in measured)


def validate_reciprocal_scale(scale):
    if not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0 or scale > 2**20 or scale & (scale - 1):
        raise ValueError("reciprocal scale must be a power of two in [1, 2**20]")
    return scale


def scaled_standard_attention(q, k, v, reciprocal_scale):
    """CrypTen softmax with public denominator scaling; not the native path."""
    import crypten
    scale = validate_reciprocal_scale(reciprocal_scale)
    scores = q.matmul(k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    maximum = scores.max(dim=-1, keepdim=True)[0]
    numerator = (scores - maximum).exp()
    denominator = numerator.sum(dim=-1, keepdim=True)
    with crypten.cfg.temp_override({"functions.reciprocal_all_pos": True}):
        inverse = denominator.div(scale).reciprocal().div(scale)
    return (numerator * inverse).matmul(v)


class AuditedPlainOperators(PlainStandardOperators):
    def __init__(self, bound=None):
        self.bound = bound
        self.coefficients = _chebyshev_coefficients(16, bound) if bound is not None else None
        self.gelu_max_abs = []

    def gelu(self, x):
        self.gelu_max_abs.append(float(x.abs().max()))
        return chebyshev_gelu(x, self.bound, self.coefficients) if self.coefficients else super().gelu(x)


class CrypTenNativeOperators(SecureOperators):
    def __init__(self, gelu_method="erf", chebyshev_bound=12.0,
                 softmax_method="native", reciprocal_scale=256):
        # Public conv, CLS and linear methods need no Ring3 runtime.
        if gelu_method not in ("erf", "chebyshev16"):
            raise ValueError("unknown GeLU method")
        if not math.isfinite(chebyshev_bound) or chebyshev_bound <= 0:
            raise ValueError("Chebyshev bound must be finite and positive")
        self.gelu_method = gelu_method
        self.chebyshev_bound = chebyshev_bound
        self.coefficients = _chebyshev_coefficients(16, chebyshev_bound) if gelu_method == "chebyshev16" else None
        if softmax_method not in ("native", "scaled"):
            raise ValueError("unknown softmax method")
        self.softmax_method = softmax_method
        self.reciprocal_scale = validate_reciprocal_scale(reciprocal_scale)

    def norm(self, x, layer):
        return crypten_native_layernorm(x, layer.weight.detach(), layer.bias.detach(), layer.eps)

    def gelu(self, x):
        if self.gelu_method == "chebyshev16":
            return chebyshev_gelu(x, self.chebyshev_bound, self.coefficients)
        return crypten_erf_gelu(x)

    def attention(self, q, k, v, module):
        if self.softmax_method == "scaled":
            return scaled_standard_attention(q, k, v, self.reciprocal_scale)
        return standard_attention(q, k, v)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    if options["sqrt_iters"] is not None:
        crypten.cfg.functions.sqrt_nr_iters = options["sqrt_iters"]
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    material_cache = StrictProviderCache(crypten) if options["online_only"] else None
    if material_cache:
        material_cache.install()
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"], None)
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        with torch.no_grad():
            reference_ops = AuditedPlainOperators()
            reference = forward_vit(model, image, reference_ops).double()
            cheb_audit = None
            if options["gelu"] == "chebyshev16":
                cheb_ops = AuditedPlainOperators(options["chebyshev_bound"])
                cheb_reference = forward_vit(model, image, cheb_ops).double()
                cheb_error = (cheb_reference - reference).abs()
                cheb_audit = {"bound": options["chebyshev_bound"],
                    "exact_gelu_input_max_abs_by_layer": reference_ops.gelu_max_abs,
                    "chebyshev_gelu_input_max_abs_by_layer": cheb_ops.gelu_max_abs,
                    "within_bound": all(x <= options["chebyshev_bound"] for x in
                        reference_ops.gelu_max_abs + cheb_ops.gelu_max_abs),
                    "float_logits_mae_vs_exact": float(cheb_error.mean()),
                    "float_logits_max_abs_vs_exact": float(cheb_error.max()),
                    "scope": "public image and plaintext model only; secret intermediates are not checked"}
                if rank == 0:
                    print(json.dumps({"event": "public_chebyshev_range_audit", **cheb_audit}), flush=True)
                if not cheb_audit["within_bound"]:
                    raise ValueError("public plaintext GeLU input exceeds Chebyshev bound")
        ops = CrypTenNativeOperators(options["gelu"], options["chebyshev_bound"],
                                     options["softmax"], options["reciprocal_scale"])
        records = []
        for iteration in range(options["repeats"] + 1):
            preprocessing = None
            if material_cache and iteration:
                _, preprocessing = _crypten_measure(device, crypten, material_cache.prepare)
            shared, sharing = _crypten_measure(device, crypten, lambda: crypten.cryptensor(
                image, src=0, precision=options["precision_bits"], device=device))
            if material_cache:
                material_cache.begin_trace() if iteration == 0 else material_cache.begin_replay()
            torch.cuda.reset_peak_memory_stats(device)
            def forward():
                def progress(index, value):
                    if rank == 0:
                        print(json.dumps({"event": "layer_complete", "warmup": iteration == 0,
                                          "layer": index + 1}), flush=True)
                truncation = (experimental_truncation(options["truncation"])
                              if options["truncation"] != "legacy" else nullcontext())
                with torch.no_grad(), truncation:
                    return forward_vit(model, shared, ops, on_layer=progress)
            output, online = _crypten_measure(device, crypten, forward)
            if material_cache:
                material_cache.end_trace() if iteration == 0 else material_cache.end_replay()
            opened, reconstruction = _crypten_measure(device, crypten, lambda: reveal_exact(output))
            actual = _plain(opened)
            finite = bool(torch.isfinite(actual).all())
            diff = (actual - reference).abs()
            own_reference = cheb_reference if cheb_audit is not None else reference
            own_diff = (actual - own_reference).abs()
            errors = {"finite": finite, "mae": float(diff.mean()) if finite else None,
                      "max_abs": float(diff.max()) if finite else None,
                      "plaintext_top1": int(reference.argmax(-1).item()),
                      "secure_top1": int(actual.argmax(-1).item()) if finite else None}
            own_errors = {"finite": finite, "mae": float(own_diff.mean()) if finite else None,
                          "max_abs": float(own_diff.max()) if finite else None,
                          "plaintext_top1": int(own_reference.argmax(-1).item()),
                          "secure_top1": int(actual.argmax(-1).item()) if finite else None}
            row = {"warmup": iteration == 0, "input_sharing": sharing,
                   "forward_wall": online, "online_crypten": online,
                   "output_reconstruction": reconstruction,
                   "error_vs_plaintext_logits": errors,
                   "error_vs_own_formula_logits": own_errors}
            if material_cache:
                row.update(crypten_preprocessing=preprocessing,
                           precomputed_provider_requests=len(material_cache.requests))
            if rank == 0:
                records.append(row)
                print(json.dumps({"event": "iteration_complete", "record": row}), flush=True)
        if rank == 0:
            passed = numerical_gate(records, options["max_abs_error"], include_warmup=True)
            report = {"status": "benchmark_passed_not_security_validated" if passed else "benchmark_failed_correctness",
                      "backend": "gloo", "world_size": 3,
                      "output_decoding": "signed raw integer divided by encoder scale in float64",
                      "scope": "full ViTLite forward; public weights and public image shared as secret input; batch 1",
                      "method": f"CrypTen softmax ({options['softmax']}), {options['gelu']} GeLU and sqrt/reciprocal LayerNorm",
                      "softmax_method": options["softmax"],
                      "truncation_method": options["truncation"],
                      "preprocessing_mode": "precomputed_strict" if material_cache else "inline",
                      "truncation_note": "non-legacy modes use unreviewed secret share-wrap counting; not CrypTen native or security validated",
                      "reciprocal_scale": options["reciprocal_scale"] if options["softmax"] == "scaled" else None,
                      "softmax_note": "scaled is a CrypTen adaptation using 1/(d/S)/S; native uses CrypTen softmax unchanged",
                      "gelu_method": options["gelu"], "public_chebyshev_range_audit": cheb_audit,
                      "gelu_note": ("Chebyshev is a CrypTen adaptation with reconstructed coefficients, not CrypTen native or original CryptGPU timing."
                                    if cheb_audit is not None else "CrypTen native Taylor-erf approximation."),
                      "model": metadata, "weights_note": "MPCViT checkpoint weights; attention replaced by softmax without retraining, not a trained standard ViT",
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "repeats": options["repeats"],
                      "sqrt_nr_iters": crypten.cfg.functions.sqrt_nr_iters,
                      "max_abs_error_threshold": options["max_abs_error"], "records": records,
                      "correctness_gate": "all iterations including warmup must meet max error against selected plaintext formula; exact GeLU model deviation reported separately",
                      "crypten_import": crypten.__file__, "python": platform.python_version(),
                      "torch": torch.__version__,
                      "timing_note": ("All traced CrypTen provider materials precomputed before input sharing; online forward uses strict single-use replay. Input sharing and reconstruction are separate."
                                      if material_cache else "CrypTen provider work is included in forward; input sharing and output reconstruction are separate."),
                      "communication_note": "CrypTen bidirectional communicator logs divided by two; collective bytes are estimates.",
                      "security_note": "Numerical execution only; comparison with experimental Ring3 is not a proof of equivalent security."}
            out = Path(options["output"])
            out.mkdir(parents=True, exist_ok=False)
            with (out / "comparison.json").open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write("\n")
            print(json.dumps({"report": str(out / "comparison.json"), "status": report["status"]}), flush=True)
    finally:
        if material_cache:
            material_cache.close()
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision-bits", type=int, default=20, choices=(17, 20))
    parser.add_argument("--sqrt-iters", type=int, help="override CrypTen inverse-square-root iterations")
    parser.add_argument("--softmax", choices=("native", "scaled"), default="native")
    parser.add_argument("--truncation", choices=("legacy", "experimental", "batched", "carry_save"), default="legacy",
                        help="experimental binary share-wrap counting is unreviewed and memory intensive")
    parser.add_argument("--reciprocal-scale", type=int, default=256,
                        help="public power-of-two scale used by adapted softmax")
    parser.add_argument("--gelu", choices=("erf", "chebyshev16"), default="erf")
    parser.add_argument("--chebyshev-bound", type=float, default=12.0,
                        help="fit 16-degree interpolant on [-bound, bound]; public audit only")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--online-only", action="store_true",
                        help="precompute fresh provider materials before input sharing; strict sequence replay may use substantial GPU memory")
    parser.add_argument("--max-abs-error", type=float, default=1.0)
    args = parser.parse_args()
    try:
        validate_reciprocal_scale(args.reciprocal_scale)
    except ValueError as error:
        parser.error(str(error))
    if (args.repeats < 1 or not 0 < args.max_abs_error < float("inf") or
            not math.isfinite(args.chebyshev_bound) or args.chebyshev_bound <= 0 or
            (args.sqrt_iters is not None and not 1 <= args.sqrt_iters <= 50)):
        parser.error("repeats and max-abs-error must be positive")
    if Path(args.output).exists():
        parser.error("output already exists")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-vit-native-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
