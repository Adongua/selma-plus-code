"""Three-party CUDA/NCCL microbenchmark for hAttention Ruler-sReLU."""

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide

from .srelu import SReLUProfile, prepare_srelu, ruler_srelu


def _parse_shape(value):
    try:
        shape = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must use x-separated integers") from error
    if not shape or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def _measure(base, function):
    dist.barrier(device_ids=[base.device.index])
    torch.cuda.synchronize(base.device)
    bytes_before = base.sent_bytes
    phases_before = base.phases
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize(base.device)
    seconds = time.perf_counter() - started
    local = torch.tensor(
        [seconds, base.sent_bytes - bytes_before, base.phases - phases_before],
        dtype=torch.float64,
        device=base.device,
    )
    gathered = [torch.empty_like(local) for _ in range(3)]
    dist.all_gather(gathered, local)
    rows = torch.stack(gathered).cpu()
    return result, {
        "seconds_max": rows[:, 0].max().item(),
        "sent_bytes_sum": int(rows[:, 1].sum().item()),
        "exchange_phases_per_rank": rows[:, 2].to(torch.int64).tolist(),
    }


def _error_metrics(actual, expected):
    difference = (actual - expected).abs().to(torch.float64)
    return {
        "mae": difference.mean().item(),
        "max_abs": difference.max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "p99_abs": torch.quantile(difference.reshape(-1), 0.99).item(),
    }


def _worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    try:
        base = CudaRing3(device=device, reveal_mode=options["reveal_mode"])
        wide = CudaRing3Wide(base)
        profile = SReLUProfile(
            radius=options["radius"],
            input_bound=options["input_bound"],
        )
        shape = tuple(options["shape"])
        generator = torch.Generator().manual_seed(options["seed"])
        strict_bound = options["input_bound"] - 1.0 / profile.input_scale
        plaintext = (
            options["input_std"]
            * torch.randn(shape, generator=generator, dtype=torch.float64)
        ).clamp(-strict_bound, strict_bound).to(device)
        encoded = profile.encode(plaintext)
        quantized = profile.decode(encoded)
        shifted_reference = profile.shifted_reference(quantized)
        corrected_reference = profile.corrected_reference(quantized)
        fixed_point_reference = profile.fixed_point_corrected_reference(quantized)
        relu_reference = torch.relu(quantized)

        records = []
        for iteration in range(options["repeats"] + 1):
            share, input_sharing = _measure(
                base,
                lambda: base.private(encoded, owner=0),
            )
            material, offline = _measure(
                base,
                lambda: prepare_srelu(base, wide, shape, profile),
            )
            output, online = _measure(
                base,
                lambda: ruler_srelu(base, wide, share, material),
            )
            opened, reconstruction = _measure(base, lambda: base.reveal(output))
            if rank == 0:
                actual = profile.decode(opened)
                records.append(
                    {
                        "warmup": iteration == 0,
                        "input_sharing": input_sharing,
                        "offline": offline,
                        "online": online,
                        "output_reconstruction": reconstruction,
                        "error_vs_corrected_srelu_reference": _error_metrics(
                            actual, corrected_reference
                        ),
                        "error_vs_fixed_point_reference": _error_metrics(
                            actual, fixed_point_reference
                        ),
                        "error_vs_shifted_srelu_reference": _error_metrics(
                            actual
                            - profile.half_radius_q17 / profile.input_scale,
                            shifted_reference,
                        ),
                        "error_vs_relu": _error_metrics(actual, relu_reference),
                        "output_min": actual.min().item(),
                        "output_max": actual.max().item(),
                    }
                )

        if rank == 0:
            output_directory = Path(options["output"])
            output_directory.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "framework": "custom Ring3 / NCCL / CUDA",
                "operator": "shifted Ruler-sReLU followed by public r/2 correction",
                "world_size": 3,
                "shape": list(shape),
                "radius": profile.radius,
                "repeats": options["repeats"],
                "reveal_mode": options["reveal_mode"],
                "fixed_point": {
                    "input_output": "Q17",
                    "radicand": "Q16",
                    "reciprocal_scale": "Q50",
                    "radius_squared_q16": profile.radius_squared_q16,
                    "radius_q17": profile.radius_q17,
                    "half_radius_q17": profile.half_radius_q17,
                },
                "fixture": "synthetic normal generated by CPU RNG before timing",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "communication": "sum of Ring3 sent payload bytes across three parties; receive bytes are not double-counted",
                "records": records,
                "security": "Ruler masked openings are not privacy validated; CUDA Philox is a benchmark PRG, not a reviewed CSPRNG",
            }
            report_path = output_directory / "comparison.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            last = records[-1]
            print(
                json.dumps(
                    {
                        "report": str(report_path),
                        "records": len(records),
                        "online_seconds": last["online"]["seconds_max"],
                        "online_bytes": last["online"]["sent_bytes_sum"],
                        "online_phases_per_rank": last["online"][
                            "exchange_phases_per_rank"
                        ],
                        "mae_vs_corrected_srelu": last[
                            "error_vs_corrected_srelu_reference"
                        ]["mae"],
                        "mae_vs_fixed_point_reference": last[
                            "error_vs_fixed_point_reference"
                        ]["mae"],
                        "max_abs_vs_corrected_srelu": last[
                            "error_vs_corrected_srelu_reference"
                        ]["max_abs"],
                        "output_min": last["output_min"],
                    },
                    indent=2,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=_parse_shape, default=(1, 2, 17, 17))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--radius", type=float, default=0.0055)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--input-std", type=float, default=0.5)
    parser.add_argument("--input-bound", type=float, default=16.0)
    parser.add_argument(
        "--reveal-mode",
        choices=("all_reduce", "all_gather", "p2p"),
        default="p2p",
    )
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    if args.radius <= 0 or not math.isfinite(args.radius):
        parser.error("radius must be positive and finite")
    if args.input_std <= 0 or not math.isfinite(args.input_std):
        parser.error("input-std must be positive and finite")
    if args.input_bound <= 0 or not math.isfinite(args.input_bound):
        parser.error("input-bound must be positive and finite")
    if torch.cuda.device_count() < 3:
        parser.error("at least three visible CUDA devices are required")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    with tempfile.TemporaryDirectory(
        prefix="hattention-srelu-cuda-", dir=args.output.parent
    ) as directory:
        mp.spawn(
            _worker,
            args=(Path(directory, "store").as_uri(), options),
            nprocs=3,
            join=True,
        )


if __name__ == "__main__":
    main()
