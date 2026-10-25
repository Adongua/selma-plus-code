"""Three-party CUDA/NCCL microbenchmark for the custom sGeLU protocol."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import platform
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from .protocol import prepare_g2, ruler_g2
from .runtime_cuda import CudaRing3, CudaRing3Wide
from .sgelu import MU0, error_metrics, g2


def measure(base, fn):
    dist.barrier()
    torch.cuda.synchronize(base.device)
    sent_before, phases_before = base.sent_bytes, base.phases
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(base.device)
    seconds = time.perf_counter() - started
    metrics = torch.tensor([seconds, base.sent_bytes - sent_before,
                            base.phases - phases_before],
                           dtype=torch.float64, device=base.device)
    gathered = [torch.empty_like(metrics) for _ in range(3)]
    dist.all_gather(gathered, metrics)
    rows = torch.stack(gathered).cpu()
    return result, {"seconds_max": rows[:, 0].max().item(),
                    "sent_bytes_sum": int(rows[:, 1].sum()),
                    "exchange_phases_per_rank": rows[:, 2].int().tolist()}


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank,
                            world_size=3, timeout=timedelta(seconds=600),
                            device_id=device)
    try:
        base, wide = CudaRing3(device=device), None
        wide = CudaRing3Wide(base)
        generator = torch.Generator(device=device).manual_seed(options["seed"])
        x = (1.5 * torch.randn((options["batch"], options["count"]),
                               generator=generator, dtype=torch.float64,
                               device=device)).clamp(-8, 8)
        x_q17 = (x * (1 << 17)).round().long()
        target = F.gelu(x, approximate="none")
        records = []
        for iteration in range(options["repeats"] + 1):
            shares, input_metrics = measure(base, lambda: base.private(x_q17, 0))
            material, offline = measure(
                base, lambda: prepare_g2(base, wide, x_q17.shape, options["mu"]))
            output, online = measure(base, lambda: ruler_g2(base, wide, shares, material))
            opened, reconstruction = measure(base, lambda: base.reveal(output))
            if rank == 0:
                result = opened.double() / (1 << 17)
                error = error_metrics(result, target)
                records.append({"warmup": iteration == 0,
                                "input_sharing": input_metrics, "offline": offline,
                                "online": online, "output_reconstruction": reconstruction,
                                "error_vs_torch_gelu": error})
        if rank == 0:
            output = Path(options["output"])
            output.mkdir(parents=True, exist_ok=True)
            report = {"status": "executed_not_security_validated", "world_size": 3,
                      "backend": "NCCL / CUDA", "shape": [options["batch"], options["count"]],
                      "mu": options["mu"], "repeats": options["repeats"],
                      "python": platform.python_version(), "torch": torch.__version__,
                      "records": records,
                      "security": "same experimental Ruler limitations as the CPU implementation"}
            (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(output / "comparison.json"),
                              "records": len(records)}, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mu", type=float, default=MU0)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if torch.cuda.device_count() < 3:
        parser.error("at least three CUDA devices are required")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sgelu-cuda-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), options),
                 nprocs=3, join=True)


if __name__ == "__main__":
    main()
