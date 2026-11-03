"""Three-party full ViTLite forward with CrypTen-native nonlinear operators.

This uses MPCViT checkpoint weights with every attention head replaced by
standard softmax. It is an operator-cost control, not a trained standard ViT.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile

import torch
import torch.multiprocessing as mp

from sgelu.baseline import crypten_erf_gelu
from slayernorm.baselines import crypten_native_layernorm
from slayernorm.benchmark_crypten_cuda import _crypten_measure, _plain
from .benchmark_compare_cuda import standard_attention
from .benchmark_vit_cuda import SecureOperators, image_tensor, load_model
from .student_checkpoint import file_sha256
from .vit_executor import PlainOperators, forward_vit


class PlainStandardOperators(PlainOperators):
    def attention(self, q, k, v, module):
        return torch.softmax(q.matmul(k.transpose(-2, -1)) * (q.size(-1) ** -0.5), dim=-1).matmul(v)


class CrypTenNativeOperators(SecureOperators):
    def __init__(self):
        # Public conv, CLS and linear methods need no Ring3 runtime.
        pass

    def norm(self, x, layer):
        return crypten_native_layernorm(x, layer.weight.detach(), layer.bias.detach(), layer.eps)

    def gelu(self, x):
        return crypten_erf_gelu(x)

    def attention(self, q, k, v, module):
        return standard_attention(q, k, v)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"], None)
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        with torch.no_grad():
            reference = forward_vit(model, image, PlainStandardOperators()).double()
        ops = CrypTenNativeOperators()
        records = []
        for iteration in range(options["repeats"] + 1):
            shared, sharing = _crypten_measure(device, crypten, lambda: crypten.cryptensor(
                image, src=0, precision=options["precision_bits"], device=device))
            torch.cuda.reset_peak_memory_stats(device)
            def forward():
                def progress(index, value):
                    if rank == 0:
                        print(json.dumps({"event": "layer_complete", "warmup": iteration == 0,
                                          "layer": index + 1}), flush=True)
                with torch.no_grad():
                    return forward_vit(model, shared, ops, on_layer=progress)
            output, online = _crypten_measure(device, crypten, forward)
            opened, reconstruction = _crypten_measure(device, crypten, lambda: output.get_plain_text())
            actual = _plain(opened)
            finite = bool(torch.isfinite(actual).all())
            diff = (actual - reference).abs()
            errors = {"finite": finite, "mae": float(diff.mean()) if finite else None,
                      "max_abs": float(diff.max()) if finite else None,
                      "plaintext_top1": int(reference.argmax(-1).item()),
                      "secure_top1": int(actual.argmax(-1).item()) if finite else None}
            row = {"warmup": iteration == 0, "input_sharing": sharing,
                   "forward_wall": online, "online_crypten": online,
                   "output_reconstruction": reconstruction,
                   "error_vs_plaintext_logits": errors}
            if rank == 0:
                records.append(row)
                print(json.dumps({"event": "iteration_complete", "record": row}), flush=True)
        if rank == 0:
            measured = [r for r in records if not r["warmup"]]
            passed = all(r["error_vs_plaintext_logits"]["finite"] and
                         r["error_vs_plaintext_logits"]["max_abs"] <= options["max_abs_error"]
                         for r in measured)
            report = {"status": "benchmark_passed_not_security_validated" if passed else "benchmark_failed_correctness",
                      "backend": "gloo", "world_size": 3,
                      "scope": "full ViTLite forward; public weights and public image shared as secret input; batch 1",
                      "method": "CrypTen native softmax, erf GeLU and sqrt/reciprocal LayerNorm",
                      "model": metadata, "weights_note": "MPCViT checkpoint weights; attention replaced by softmax without retraining, not a trained standard ViT",
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "repeats": options["repeats"],
                      "max_abs_error_threshold": options["max_abs_error"], "records": records,
                      "crypten_import": crypten.__file__, "python": platform.python_version(),
                      "torch": torch.__version__,
                      "timing_note": "CrypTen provider work is included in forward; input sharing and output reconstruction are separate.",
                      "communication_note": "CrypTen bidirectional communicator logs divided by two; collective bytes are estimates.",
                      "security_note": "Numerical execution only; comparison with experimental Ring3 is not a proof of equivalent security."}
            out = Path(options["output"])
            out.mkdir(parents=True, exist_ok=False)
            with (out / "comparison.json").open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write("\n")
            print(json.dumps({"report": str(out / "comparison.json"), "status": report["status"]}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision-bits", type=int, default=20, choices=(17, 20))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-abs-error", type=float, default=1.0)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.max_abs_error < float("inf"):
        parser.error("repeats and max-abs-error must be positive")
    if Path(args.output).exists():
        parser.error("output already exists")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-vit-native-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
