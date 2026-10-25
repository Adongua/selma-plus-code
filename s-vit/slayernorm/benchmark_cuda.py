"""Three-party CUDA/NCCL microbenchmark for the experimental sLayerNorm.

Run with one process per GPU, for example:
    CUDA_VISIBLE_DEVICES=0,1,2 python -m slayernorm.benchmark_cuda \
        --shapes 1x197x768 --repeats 3 --output /path/to/output

This benchmark covers the custom sLayerNorm methods only.  CrypTen GPU
baselines are intentionally kept separate until their communicator and
precision configuration has been independently verified.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from .protocol import METHODS, Profile, prepare, layernorm


def fixture(shape, distribution, seed, device):
    rng = torch.Generator(device=device).manual_seed(seed)
    if distribution == "normal":
        x = torch.randn(shape, generator=rng, dtype=torch.float64,
                        device=device).clamp(-4, 4)
    elif distribution == "uniform":
        x = torch.rand(shape, generator=rng, dtype=torch.float64,
                       device=device) * 8 - 4
    else:
        x = torch.full(shape, 1.25, dtype=torch.float64, device=device)
    gamma = torch.linspace(.75, 1.25, shape[-1], dtype=torch.float64, device=device)
    beta = torch.linspace(-.1, .1, shape[-1], dtype=torch.float64, device=device)
    encode = lambda v: (v * 2**17).round().long()
    return tuple(encode(v) for v in (x, gamma, beta))


def measure(base, fn):
    dist.barrier()
    torch.cuda.synchronize(base.device)
    sent, phases = base.sent_bytes, base.phases
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(base.device)
    seconds = time.perf_counter() - started
    metrics = torch.tensor([seconds, base.sent_bytes - sent, base.phases - phases],
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
        profile = Profile(epsilon=options["epsilon"], scale=options["scale"])
        records = []
        for shape in options["shapes"]:
            profile.validate(shape)
            for distribution in options["distributions"]:
                x, gamma, beta = fixture(shape, distribution, options["seed"], device)
                xf, gf, bf = [v.double() / 2**17 for v in (x, gamma, beta)]
                target = F.layer_norm(xf, (shape[-1],), gf, bf, profile.epsilon)
                for iteration in range(options["repeats"] + 1):
                    for offset in range(len(options["methods"])):
                        method = options["methods"][(iteration + offset) % len(options["methods"])]
                        shares, input_metrics = measure(base, lambda: (
                            base.private(x, 0), base.private(gamma, 1), base.private(beta, 2)))
                        material, offline = measure(
                            base, lambda: prepare(base, wide, shape, profile, method))
                        output, online = measure(
                            base, lambda: layernorm(base, wide, *shares, material))
                        opened, reconstruction = measure(base, lambda: base.reveal(output))
                        if rank == 0:
                            result = opened.double() / 2**17
                            delta = (result - target).abs()
                            records.append({
                                "shape": list(shape), "distribution": distribution,
                                "method": method, "warmup": iteration == 0,
                                "input_sharing": input_metrics, "offline": offline,
                                "online": online, "output_reconstruction": reconstruction,
                                "error_vs_torch": {
                                    "mae": delta.mean().item(),
                                    "max_abs": delta.max().item(),
                                    "rmse": delta.square().mean().sqrt().item(),
                                },
                            })
        if rank == 0:
            output = Path(options["output"])
            output.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "world_size": 3, "backend": "NCCL / CUDA",
                "devices": [f"cuda:{i}" for i in range(3)],
                "python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "repeats": options["repeats"],
                "records": records,
                "communication": "sum of NCCL point-to-point sent payload bytes across three parties",
                "security": "same experimental Ruler limitations as the CPU implementation",
            }
            (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(output / "comparison.json"),
                              "records": len(records)}, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=["1x197x768"])
    parser.add_argument("--distributions", nargs="+", choices=["normal", "uniform", "constant"],
                        default=["normal"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["fused_ruler"])
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    args.shapes = [tuple(int(n) for n in shape.split("x")) for shape in args.shapes]
    if torch.cuda.device_count() < 3:
        parser.error("at least three CUDA devices are required")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slayernorm-cuda-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), options),
                 nprocs=3, join=True)


if __name__ == "__main__":
    main()
