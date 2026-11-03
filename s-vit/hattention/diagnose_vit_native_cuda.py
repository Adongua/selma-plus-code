"""Public-fixture per-stage diagnosis for the CrypTen softmax ViT control.

All three parties open intermediate tensors. This is a numerical diagnosis,
not a private execution or performance benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.multiprocessing as mp

from slayernorm.benchmark_crypten_cuda import _plain
from .benchmark_vit_cuda import image_tensor, load_model
from .benchmark_vit_native_cuda import AuditedPlainOperators, CrypTenNativeOperators
from .diagnose_vit_cuda import error_metrics
from .student_checkpoint import file_sha256
from .vit_executor import forward_vit


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.init(device=device)
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"], None)
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        bound = options["chebyshev_bound"]
        reference_ops = AuditedPlainOperators(bound)
        references = {}
        def remember(index, site, value):
            references[(index, site)] = value.detach().clone()
        with torch.no_grad():
            reference = forward_vit(model, image, reference_ops, on_stage=remember)
        if any(x > bound for x in reference_ops.gelu_max_abs):
            raise ValueError("public plaintext Chebyshev input exceeds the fit interval")
        if rank == 0:
            print(json.dumps({"event": "public_chebyshev_range_audit", "bound": bound,
                              "input_max_abs_by_layer": reference_ops.gelu_max_abs}), flush=True)
        ops = CrypTenNativeOperators("chebyshev16", bound)
        encrypted = crypten.cryptensor(image, src=0, precision=options["precision_bits"], device=device)
        rows = []
        def inspect(index, site, value):
            # All ranks open each intermediate in the same order.
            opened = _plain(value.get_plain_text())
            if rank == 0:
                row = {"layer": index + 1, "site": site,
                       **error_metrics(opened, references[(index, site)])}
                if site == "gelu_input":
                    row["chebyshev_bound"] = bound
                    row["opened_input_exceeds_bound"] = (row["actual_max_abs"] > bound
                                                          if row["finite"] else None)
                rows.append(row)
                print(json.dumps({"event": "site_complete", **row}, allow_nan=False), flush=True)
        with torch.no_grad():
            logits = forward_vit(model, encrypted, ops, on_stage=inspect)
            actual = _plain(logits.get_plain_text())
        if rank == 0:
            final = error_metrics(actual, reference)
            if final["finite"]:
                final.update(plaintext_top1=int(reference.argmax(-1).item()),
                             secure_top1=int(actual.argmax(-1).item()))
            report = {"status": "diagnostic_completed_not_security_validated",
                      "scope": "public image and public model; intermediate activations opened",
                      "method": "CrypTen softmax + reconstructed Chebyshev16 GeLU + CrypTen LayerNorm",
                      "model": metadata, "image": str(Path(options["image"]).resolve()),
                      "image_sha256": file_sha256(options["image"]),
                      "precision": f"Q{options['precision_bits']}", "chebyshev_bound": bound,
                      "plaintext_gelu_input_max_abs_by_layer": reference_ops.gelu_max_abs,
                      "sites": rows, "logits": final,
                      "interpretation": "Compared with the same plaintext Chebyshev model; each site includes errors propagated from preceding sites.",
                      "timing": "Not a benchmark; opening intermediates changes execution and communication.",
                      "security": "Intermediate activations are opened; use public fixtures only."}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            with (root / "diagnosis.json").open("x", encoding="utf-8") as output:
                json.dump(report, output, indent=2, allow_nan=False)
                output.write("\n")
            print(json.dumps({"report": str(root / "diagnosis.json"), "logits": final}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--image", required=True, help="one public test image")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--precision-bits", type=int, choices=(17, 20), default=20)
    parser.add_argument("--chebyshev-bound", type=float, default=12.0)
    args = parser.parse_args()
    if not math.isfinite(args.chebyshev_bound) or args.chebyshev_bound <= 0:
        parser.error("Chebyshev bound must be finite and positive")
    if Path(args.output).exists():
        parser.error("output already exists")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-native-diagnose-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
