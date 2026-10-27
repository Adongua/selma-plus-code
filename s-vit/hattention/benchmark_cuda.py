"""Three-party CUDA/NCCL microbenchmark for secure LSA and Mixer heads.

This benchmark covers the linear subset of hAttention.  Secure RSA is omitted
until its sReLU and row-reciprocal protocols have independent correctness and
privacy tests.
"""

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

from sgelu.runtime_cuda import CudaRing3

from .config import infer_patch_grid, normalize_head_types
from .protocol import (
    FixedPointProfile,
    prepare_grouped_lsa_mixer,
    secure_grouped_lsa_mixer,
)
from .reference import grouped_hattention


def _parse_shape(value):
    try:
        shape = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be BxHxNxD") from error
    if len(shape) != 4 or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("shape must contain four positive integers")
    return shape


def _parse_grid(value):
    try:
        grid = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("patch grid must be ROWSxCOLS") from error
    if len(grid) != 2 or any(item <= 0 for item in grid):
        raise argparse.ArgumentTypeError("patch grid must contain two positive integers")
    return grid


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
        profile = FixedPointProfile(
            fraction_bits=options["fraction_bits"],
            coefficient_fraction_bits=options["coefficient_fraction_bits"],
            input_bound=options["input_bound"],
        )
        shape = tuple(options["shape"])
        head_types = tuple(options["head_types"])
        rho = options["rho"] if options["rho"] is not None else shape[-1] ** -0.5
        patch_grid = (
            tuple(options["patch_grid"])
            if options["patch_grid"] is not None
            else infer_patch_grid(shape[2], has_cls_token=options["has_cls_token"])
        )

        generator = torch.Generator().manual_seed(options["seed"])
        plaintext = []
        for _ in range(3):
            value = (
                options["input_std"]
                * torch.randn(shape, generator=generator, dtype=torch.float64)
            ).clamp(-options["input_bound"], options["input_bound"])
            plaintext.append(value.to(device))
        q, k, v = plaintext
        q_encoded = profile.encode(q)
        k_encoded = profile.encode(k)
        v_encoded = profile.encode(v)
        q_reference = profile.decode(q_encoded)
        k_reference = profile.decode(k_encoded)
        v_reference = profile.decode(v_encoded)
        reference = grouped_hattention(
            q_reference,
            k_reference,
            v_reference,
            head_types=head_types,
            rho=rho,
            patch_grid=patch_grid,
            has_cls_token=options["has_cls_token"],
        )

        records = []
        for iteration in range(options["repeats"] + 1):
            (q_share, k_share, v_share), input_sharing = _measure(
                base,
                lambda: (
                    base.private(q_encoded, owner=0),
                    base.private(k_encoded, owner=0),
                    base.private(v_encoded, owner=0),
                ),
            )
            material, offline = _measure(
                base,
                lambda: prepare_grouped_lsa_mixer(
                    base,
                    shape,
                    head_types,
                    rho=rho,
                    profile=profile,
                ),
            )
            output, online = _measure(
                base,
                lambda: secure_grouped_lsa_mixer(
                    base,
                    q_share,
                    k_share,
                    v_share,
                    material,
                    patch_grid=patch_grid,
                    has_cls_token=options["has_cls_token"],
                ),
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
                        "error_vs_quantized_plaintext_reference": _error_metrics(
                            actual, reference
                        ),
                    }
                )

        if rank == 0:
            output_directory = Path(options["output"])
            output_directory.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "framework": "custom Ring3 / NCCL / CUDA",
                "scope": "secure LSA plus communication-free public Mixer; RSA excluded",
                "world_size": 3,
                "shape_b_h_n_d": list(shape),
                "head_types": list(head_types),
                "patch_grid": list(patch_grid),
                "has_cls_token": options["has_cls_token"],
                "rho": rho,
                "repeats": options["repeats"],
                "reveal_mode": options["reveal_mode"],
                "fixed_point": {
                    "fraction_bits": profile.fraction_bits,
                    "coefficient_fraction_bits": profile.coefficient_fraction_bits,
                    "input_bound": profile.input_bound,
                },
                "fixture": "synthetic normal generated by CPU RNG before timing",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "communication": "sum of Ring3 sent payload bytes across three parties; receive bytes are not double-counted",
                "integer_matmul": "native torch int64 matmul with an on-device blocked multiply-reduce fallback for unsupported CUDA builds",
                "records": records,
                "security": "experimental Ring3 implementation; CUDA Philox benchmark PRG is not a reviewed CSPRNG",
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
                        "mae": last["error_vs_quantized_plaintext_reference"]["mae"],
                        "max_abs": last["error_vs_quantized_plaintext_reference"][
                            "max_abs"
                        ],
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
    parser.add_argument("--shape", type=_parse_shape, default=(1, 4, 17, 16))
    parser.add_argument(
        "--head-types",
        nargs="+",
        default=("lsa", "lsa", "mixer", "mixer"),
        help="one public path per head; this phase accepts lsa and mixer only",
    )
    parser.add_argument("--patch-grid", type=_parse_grid)
    parser.add_argument(
        "--no-cls-token",
        action="store_false",
        dest="has_cls_token",
        help="treat every token as a patch token",
    )
    parser.set_defaults(has_cls_token=True)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--input-std", type=float, default=0.25)
    parser.add_argument("--input-bound", type=float, default=8.0)
    parser.add_argument("--fraction-bits", type=int, default=17)
    parser.add_argument("--coefficient-fraction-bits", type=int, default=30)
    parser.add_argument(
        "--reveal-mode",
        choices=("all_reduce", "all_gather", "p2p"),
        default="p2p",
    )
    args = parser.parse_args()

    try:
        args.head_types = normalize_head_types(args.head_types, args.shape[1])
    except ValueError as error:
        parser.error(str(error))
    if "rsa" in args.head_types:
        parser.error("secure RSA is not implemented in this benchmark phase")
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    if args.input_std <= 0:
        parser.error("input-std must be positive")
    if args.input_bound <= 0 or not math.isfinite(args.input_bound):
        parser.error("input-bound must be positive and finite")
    if args.rho is not None and not math.isfinite(args.rho):
        parser.error("rho must be finite")
    if torch.cuda.device_count() < 3:
        parser.error("at least three visible CUDA devices are required")
    if args.patch_grid is None:
        try:
            infer_patch_grid(args.shape[2], has_cls_token=args.has_cls_token)
        except ValueError as error:
            parser.error(str(error))
    else:
        patch_count = args.shape[2] - int(args.has_cls_token)
        if args.patch_grid[0] * args.patch_grid[1] != patch_count:
            parser.error(
                f"patch-grid contains {args.patch_grid[0] * args.patch_grid[1]} "
                f"positions but the input has {patch_count} patch tokens"
            )

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options["output"] = str(args.output)
    with tempfile.TemporaryDirectory(
        prefix="hattention-cuda-", dir=args.output.parent
    ) as directory:
        mp.spawn(
            _worker,
            args=(Path(directory, "store").as_uri(), options),
            nprocs=3,
            join=True,
        )


if __name__ == "__main__":
    main()
