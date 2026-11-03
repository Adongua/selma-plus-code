"""Open public-fixture intermediate values to localize full-ViT error.

This is an invasive numerical diagnosis, not a private execution or benchmark.
The reference is the original plaintext fixed student, so differences include
operator approximation, fixed-point error and propagation from earlier sites.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from slayernorm.benchmark_crypten_cuda import _plain
from .benchmark_vit_cuda import SecureOperators, audit_public_fixture, image_tensor, load_model
from .student_checkpoint import file_sha256
from .vit_executor import PlainOperators, forward_vit


def error_metrics(actual, reference):
    actual, reference = actual.detach().double(), reference.detach().double()
    if actual.shape != reference.shape:
        raise ValueError("reference and reconstructed tensor shapes differ")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    result = {"shape": list(actual.shape), "finite": finite,
              "mae": None, "rmse": None, "max_abs": None,
              "reference_max_abs": None, "actual_max_abs": None}
    if finite:
        delta = actual - reference
        result.update(mae=float(delta.abs().mean()), rmse=float(delta.square().mean().sqrt()),
                      max_abs=float(delta.abs().max()),
                      reference_max_abs=float(reference.abs().max()),
                      actual_max_abs=float(actual.abs().max()))
    return result


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=device)
    try:
        model, metadata = load_model(options["checkpoint"], options["mpcvit_root"],
                                     options["fixed_student"])
        model = model.to(device).eval()
        image = image_tensor(options["image"]).to(device)
        reference, observations = audit_public_fixture(model, image, options["numeric_profile"])
        violations = [row for row in observations if not row["within_profile"]]
        if violations:
            raise ValueError(f"public fixture exceeds the numerical profile: {violations}")
        reference_sites = {}
        def remember(index, site, value):
            reference_sites[(index, site)] = value.detach().clone()
        with torch.no_grad():
            plain_logits = forward_vit(model, image, PlainOperators(), on_stage=remember)
        if not torch.allclose(plain_logits.double(), reference, atol=1e-6, rtol=1e-6):
            raise ArithmeticError("plaintext executor does not match model reference")
        base = CudaRing3(device=device, reveal_mode="p2p")
        wide = CudaRing3Wide(base)
        ops = SecureOperators(base, wide, options["mu"], options["numeric_profile"])
        shared = crypten.cryptensor(image, src=0, precision=17, device=device)
        rows = []
        def inspect(index, site, value):
            # Every rank participates in every reconstruction, in identical order.
            opened = _plain(value.get_plain_text())
            if rank == 0:
                row = {"layer": index + 1, "site": site,
                       **error_metrics(opened, reference_sites[(index, site)])}
                rows.append(row)
                print(json.dumps({"event": "site_complete", **row}), flush=True)
        with torch.no_grad():
            logits = forward_vit(model, shared, ops, on_stage=inspect)
            actual = _plain(logits.get_plain_text())
        if rank == 0:
            final = error_metrics(actual, plain_logits)
            if final["finite"]:
                final.update(plaintext_top1=int(plain_logits.argmax(-1).item()),
                             secure_top1=int(actual.argmax(-1).item()))
            report = {"status": "diagnostic_completed_not_security_validated",
                      "scope": "public image and public model; intermediate activations opened",
                      "model": metadata, "image": str(Path(options["image"]).resolve()),
                      "image_sha256": file_sha256(options["image"]),
                      "numeric_profile": options["numeric_profile"], "mu": options["mu"],
                      "public_fixture_range_audit": observations, "sites": rows,
                      "logits": final,
                      "interpretation": "Differences include earlier-layer propagation, operator approximation and fixed-point error; this does not isolate an individual operator.",
                      "timing": "Not a benchmark: intermediate reconstruction changes communication and execution.",
                      "security": "Public-fixture diagnosis opens intermediates; Ruler privacy is not validated."}
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
    parser.add_argument("--fixed-student", required=True)
    parser.add_argument("--image", required=True, help="one public test image")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--numeric-profile", choices=("legacy", "wide-range"), default="wide-range")
    parser.add_argument("--mu", type=float, default=1.4621998268680836)
    args = parser.parse_args()
    if not 0.25 <= args.mu <= 4:
        parser.error("mu must be in [0.25, 4]")
    if Path(args.output).exists():
        parser.error("output already exists")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-vit-diagnose-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
