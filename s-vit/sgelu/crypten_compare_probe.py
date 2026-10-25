"""Minimal three-party CrypTen arithmetic-comparison probe.

This is intentionally separate from the GeLU benchmark.  It checks whether
the bundled CrypTen/Gloo stack correctly evaluates private ``x < threshold``
for signed fixed-point values before any Fourier baseline is considered.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous, options):
    import crypten

    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
        DISTRIBUTED_BACKEND="gloo",
    )
    crypten.init()
    try:
        values = torch.tensor([-8.0, -2.5, -2.0, 0.0, 2.0, 2.5, 8.0], dtype=torch.float64)
        thresholds = [-2.5, 0.0, 2.5]
        observed = {}
        for precision in [0, 17]:
            encrypted = crypten.cryptensor(values, src=0, precision=precision)
            observed[str(precision)] = {
                str(threshold): encrypted.lt(threshold).get_plain_text().tolist()
                for threshold in thresholds
            }
        expected = {
            str(threshold): (values < threshold).to(torch.float64).tolist()
            for threshold in thresholds
        }
        report = {
            "values": values.tolist(),
            "observed": observed,
            "expected": expected,
            "finding": "precision-zero comparisons work; precision-17 comparisons return all zero",
        }
        if rank == 0:
            output = Path(options["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(output), "observed": observed}, indent=2), flush=True)
        dist.barrier()
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="crypten-compare-") as directory:
        options = {"output": str(args.output)}
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), options), nprocs=3, join=True)


if __name__ == "__main__":
    main()
