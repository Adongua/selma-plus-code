"""Measure 3PC truncation variants with public fixtures, not a privacy test."""
import argparse
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}" if options["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    from crypten.mpc.primitives import beaver
    from .experimental_crypten_truncation import experimental_truncate, batched_truncate

    crypten.cfg.encoder.precision_bits = options["precision_bits"]
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        scale = 1 << options["precision_bits"]
        # Public mixed-sign Q2p-sized raw inputs; identical across methods.
        idx = torch.arange(options["elements"], dtype=torch.int64, device=device)
        fixture = torch.where(idx % 2 == 0, float(scale), -float(scale)).to(torch.float64)
        methods = {"legacy": beaver.truncate, "experimental": experimental_truncate,
                   "batched": batched_truncate}
        records = []
        for method in options["methods"]:
            for iteration in range(options["repeats"] + 1):
                secret = crypten.cryptensor(fixture, src=0, precision=options["precision_bits"], device=device)
                opened = secret.reveal()
                before = opened.tensor() if hasattr(opened, "tensor") else opened
                expected = torch.div(before, scale, rounding_mode="trunc")
                dist.barrier()
                crypten.reset_communication_stats()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                methods[method](secret._tensor, scale)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                stats = crypten.communicator.get().get_communication_stats()
                peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                local = torch.tensor([elapsed, stats["bytes"], stats["rounds"], peak], dtype=torch.float64)
                gathered = [torch.empty_like(local) for _ in range(3)]
                dist.all_gather(gathered, local)
                reconstructed = secret.reveal()
                actual = reconstructed.tensor() if hasattr(reconstructed, "tensor") else reconstructed
                delta = actual - expected
                max_error = int(delta.abs().max())
                errors = int((delta.abs() > 4).sum())
                if rank == 0:
                    metrics = torch.stack(gathered)
                    record = {"method": method, "warmup": iteration == 0,
                              "seconds_max": float(metrics[:, 0].max()),
                              "system_bytes_estimate": int(metrics[:, 1].sum()) // 2,
                              "rounds_per_rank": [int(x) for x in metrics[:, 2]],
                              "peak_allocated_bytes_per_rank": [int(x) for x in metrics[:, 3]] if device.type == "cuda" else None,
                              "max_raw_error": max_error, "large_error_count": errors}
                    records.append(record)
                    print(json.dumps({"event": "iteration_complete", **record}), flush=True)
        if rank == 0:
            summary = {}
            for method in options["methods"]:
                rows = [r for r in records if r["method"] == method and not r["warmup"]]
                all_rows = [r for r in records if r["method"] == method]
                summary[method] = {"mean_seconds": statistics.mean(r["seconds_max"] for r in rows),
                                   "median_seconds": statistics.median(r["seconds_max"] for r in rows),
                                   "mean_system_bytes_estimate": statistics.mean(r["system_bytes_estimate"] for r in rows),
                                   "all_including_warmup_passed": all(r["large_error_count"] == 0 for r in all_rows)}
            report = {"scope": "public fixture only; output opened after timing; not security validation",
                      "provider": type(crypten.mpc.get_default_provider()).__name__,
                      "crypten_import": crypten.__file__, "backend": "gloo", "world_size": 3,
                      "device": options["device"], "precision_bits": options["precision_bits"],
                      "elements": options["elements"], "repeats": options["repeats"],
                      "timing_note": "includes provider work; excludes input sharing, output opening, metric gathering; max over parties",
                      "communication_note": "CrypTen summed bidirectional logs divided by two; collective bytes estimated",
                      "records": records, "summary": summary}
            root = Path(options["output"])
            root.mkdir(parents=True, exist_ok=False)
            (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"report": str(root / "comparison.json"), "summary": summary}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision-bits", choices=(17, 20), type=int, default=20)
    parser.add_argument("--elements", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", choices=("legacy", "experimental", "batched"), nargs="+",
                        default=("legacy", "experimental", "batched"))
    args = parser.parse_args()
    if args.elements < 1 or args.repeats < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("elements and repeats must be positive and methods must not repeat")
    if Path(args.output).exists():
        parser.error("output already exists")
    if args.device == "cuda" and torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="crypten-truncation-bench-") as directory:
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
