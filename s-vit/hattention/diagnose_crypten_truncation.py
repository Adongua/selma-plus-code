"""Stress CrypTen's three-party truncation on public fixed-point values only.

This intentionally opens intermediates and is neither private nor a benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.multiprocessing as mp


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}" if options["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.init(device=device)
    try:
        records = []
        scale = 1 << options["precision_bits"]
        # A secret encoded value of 2**precision has raw magnitude 2**(2*precision),
        # similar to the Q2p product entering fixed-point truncation.
        value = float(1 << options["precision_bits"]) * options["sign"]
        fixture = torch.full((options["elements"],), value, device=device)
        for iteration in range(options["repeats"]):
            secret = crypten.cryptensor(fixture, src=0, precision=options["precision_bits"], device=device)
            raw_before = secret.reveal()
            raw_before = raw_before.tensor() if hasattr(raw_before, "tensor") else raw_before
            if options["mode"] == "experimental":
                from .experimental_crypten_truncation import experimental_truncate
                experimental_truncate(secret._tensor, scale)
            else:
                secret._tensor.div_(scale)
            opened = secret.reveal()
            raw_after = opened.tensor() if hasattr(opened, "tensor") else opened
            expected = torch.div(raw_before, scale, rounding_mode="trunc")
            delta = raw_after - expected
            wrap = 1 << (64 - options["precision_bits"])
            index = int(delta.abs().argmax())
            record = {"iteration": iteration + 1,
                      "elements": options["elements"],
                      "large_errors": int((delta.abs() > 4).sum()),
                      "near_one_wrap": int(((delta.abs() - wrap).abs().le(4)).sum()),
                      "max_abs_raw_error": int(delta.abs().max()),
                      "max_signed_raw_error": int(delta[index]),
                      "expected_raw_at_max": int(expected[index]),
                      "actual_raw_at_max": int(raw_after[index])}
            if rank == 0:
                records.append(record)
                print(json.dumps({"event": "iteration_complete", **record}), flush=True)
        if rank == 0:
            report = {"scope": "public input diagnostic; intermediate values opened; not private or a benchmark",
                      "crypten_import": crypten.__file__, "backend": "gloo", "world_size": 3,
                      "device": options["device"], "precision_bits": options["precision_bits"],
                      "mode": options["mode"],
                      "divisor": scale, "raw_input": options["sign"] * scale * scale,
                      "expected_one_wrap_raw": 1 << (64 - options["precision_bits"]),
                      "records": records}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            (root / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"report": str(root / "diagnosis.json")}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision-bits", choices=(17, 20), type=int, default=20)
    parser.add_argument("--elements", type=int, default=1000000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--mode", choices=("legacy", "experimental"), default="legacy",
                        help="experimental uses unreviewed secret wrap correction and is much more expensive")
    args = parser.parse_args()
    if args.elements < 1 or args.repeats < 1:
        parser.error("--elements and --repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="crypten-truncation-") as directory:
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
