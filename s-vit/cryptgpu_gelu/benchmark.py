"""Three-party CUDA microbenchmark on the vendored CryptGPU backend."""

import argparse
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

from sgelu.baseline import puma_gelu_reference, secformer_fourier_gelu_reference
from sgelu.sgelu import MU0, error_metrics, g2

from .operators import (chebyshev16_gelu_reference,
                        cryptgpu_chebyshev16_gelu, cryptgpu_erf_gelu,
                        cryptgpu_fourier_gelu,
                        cryptgpu_puma_gelu, cryptgpu_quad_gelu,
                        prepare_fourier)
from .runtime import CryptGpuSGeLU

METHODS = ("sgelu", "chebyshev16", "puma", "quad", "erf", "fourier")


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 1x197x768") from error
    if not shape or any(dimension < 1 for dimension in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def _mean(values):
    return sum(values) / len(values)


def _write_summary(report, path):
    formal = [record for record in report["records"] if not record["warmup"]]
    methods = report["methods"]
    online_means = {
        method: _mean([
            record["methods"][method]["online"]["seconds_max"]
            for record in formal
        ])
        for method in methods
    }
    sgelu_time = online_means.get("sgelu")
    rows = [
        "# CryptGPU GeLU microbenchmark",
        "",
        f"- Shape: `{tuple(report['shape'])}`",
        f"- Backend: {report['backend']}",
        f"- Distribution: {report['distribution']}",
        f"- Formal repeats: {report['repeats']}",
        "- Values are means over formal repeats; warmup is excluded.",
        "",
        "| Method | Offline (s) | Online (s) | Online sent (MiB) | "
        "Rounds/rank | Baseline/sGeLU time | MAE vs exact GeLU | "
        "Max error vs method ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        entries = [record["methods"][method] for record in formal]
        offline = _mean([entry["offline"]["seconds_max"] for entry in entries])
        online = online_means[method]
        sent = _mean([entry["online"]["sent_bytes_sum"] for entry in entries])
        rounds = _mean([
            max(entry["online"]["rounds_per_rank"]) for entry in entries
        ])
        mae = _mean([
            entry["error_vs_torch_gelu"]["mae"] for entry in entries
        ])
        numerical = _mean([
            entry["error_vs_method_reference"]["max_abs"] for entry in entries
        ])
        ratio = online / sgelu_time if sgelu_time else float("nan")
        rows.append(
            f"| {method} | {offline:.6f} | {online:.6f} | "
            f"{sent / (1 << 20):.3f} | {rounds:.1f} | {ratio:.3f} | "
            f"{mae:.6g} | {numerical:.6g} |"
        )
    rows.extend([
        "",
        "`Baseline/sGeLU time > 1` means the baseline is slower than sGeLU; "
        "a value below 1 means it is faster.",
        "",
        "This is an engineering benchmark on the vendored CryptGPU backend, "
        "not a claim that the cited systems originally used CryptGPU.",
        "",
    ])
    path.write_text("\n".join(rows))


def _sync(device):
    torch.cuda.synchronize(device)


def _sample_input(shape, generator, distribution):
    if distribution == "normal":
        return (1.5 * torch.randn(
            shape, generator=generator, dtype=torch.float64
        )).clamp(-8, 8)
    if distribution == "uniform":
        return torch.empty(shape, dtype=torch.float64).uniform_(
            -4.0, 4.0, generator=generator
        )
    raise ValueError(f"unknown distribution {distribution}")


def _aggregate(device, seconds, communicator):
    row = torch.tensor([seconds, communicator.comm_bytes,
                        communicator.comm_rounds, communicator.comm_time],
                       dtype=torch.float64, device=device)
    rows = [torch.empty_like(row) for _ in range(3)]
    dist.all_gather(rows, row)
    values = torch.stack(rows).cpu()
    return {
        "seconds_max": values[:, 0].max().item(),
        "sent_bytes_sum": int(values[:, 1].sum().item()),
        "rounds_per_rank": values[:, 2].int().tolist(),
        "comm_time_sum": values[:, 3].sum().item(),
    }


def _measure(device, communicator, function):
    dist.barrier()
    communicator.reset_communication_stats()
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    return result, _aggregate(device, seconds, communicator)


def _prepare_method(device, encrypted, method, mu):
    if method == "sgelu":
        operator = CryptGpuSGeLU(device, mu)
        operator.prepare(encrypted.share.shape)
        return lambda: operator(encrypted.clone())
    if method == "fourier":
        material = prepare_fourier(encrypted.share.shape, device)
        return lambda: cryptgpu_fourier_gelu(encrypted.clone(), material)
    return lambda: _evaluate(encrypted.clone(), method)


def _evaluate(encrypted, method):
    if method == "puma":
        return cryptgpu_puma_gelu(encrypted)
    if method == "quad":
        return cryptgpu_quad_gelu(encrypted)
    if method == "erf":
        return cryptgpu_erf_gelu(encrypted)
    if method == "chebyshev16":
        return cryptgpu_chebyshev16_gelu(encrypted)
    raise ValueError(f"unknown method {method}")


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="nccl")

    import crypten
    import crypten.communicator as comm

    crypten.init(device=device, use_csprng=False, sync_key=False)
    communicator = comm.get()
    communicator.set_verbosity(True)
    try:
        if "CryptGPU" not in str(Path(crypten.__file__).resolve()):
            raise RuntimeError(f"wrong crypten package imported: {crypten.__file__}")
        generator = torch.Generator().manual_seed(options["seed"])
        shape = tuple(options["shape"])
        x_cpu = _sample_input(shape, generator, options["distribution"])
        x = x_cpu.to(device)
        exact = F.gelu(x, approximate="none")
        references = {
            "sgelu": g2(x, options["mu"]),
            "puma": puma_gelu_reference(x),
            "quad": 0.125 * x.square() + 0.25 * x + 0.5,
            "erf": exact,
            "fourier": secformer_fourier_gelu_reference(x),
            "chebyshev16": chebyshev16_gelu_reference(x),
        }
        records = []
        for iteration in range(options["repeats"] + 1):
            encrypted, sharing = _measure(
                device, communicator,
                lambda: crypten.cryptensor(x, src=0, precision=17, device=device),
            )
            row = {"warmup": iteration == 0, "input_sharing": sharing,
                   "methods": {}}
            for method in options["methods"]:
                evaluator, offline = _measure(
                    device, communicator,
                    lambda method=method: _prepare_method(
                        device, encrypted, method, options["mu"]),
                )
                output, online = _measure(
                    device, communicator, evaluator,
                )
                opened, reconstruction = _measure(
                    device, communicator, lambda output=output: output.get_plain_text()
                )
                if rank == 0:
                    actual = (opened.tensor() if hasattr(opened, "tensor")
                              else opened).double()
                    entry = {"offline": offline, "online": online,
                             "output_reconstruction": reconstruction,
                             "error_vs_method_reference": error_metrics(
                                 actual, references[method]),
                             "error_vs_torch_gelu": error_metrics(actual, exact)}
                    row["methods"][method] = entry
            if rank == 0:
                records.append(row)

        if rank == 0:
            output_dir = Path(options["output"])
            output_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "backend": "CryptGPU replicated-sharing 3PC / NCCL / CUDA",
                "world_size": 3,
                "shape": list(shape),
                "distribution": options["distribution"],
                "methods": options["methods"],
                "repeats": options["repeats"],
                "mu": options["mu"],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
                "crypten_source": str(Path(crypten.__file__).resolve()),
                "randomness": "PyTorch GPU PRG; benchmark-correct but not CSPRNG-secure",
                "method_notes": {
                    "sgelu": "manuscript Ruler-G2 port over CryptGPU comms; original privacy caveats remain",
                    "chebyshev16": "degree-16 Chebyshev interpolant on [-8,8]; recovered coefficients unavailable",
                    "puma": "PUMA piecewise polynomial port using CryptGPU comparisons",
                    "quad": "MPCFormer 0.125*x^2+0.25*x+0.5 port",
                    "erf": "CrypTen-style 8-iteration Taylor-erf port",
                    "fourier": "SecFormer masked-Fourier port; masked delta is opened",
                },
                "records": records,
            }
            report_path = output_dir / "comparison.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            summary_path = output_dir / "summary.md"
            _write_summary(report, summary_path)
            final = records[-1]
            print(json.dumps({
                "report": str(report_path),
                "summary": str(summary_path),
                "formal": {name: data["online"]["seconds_max"]
                           for name, data in final["methods"].items()},
            }, indent=2), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument(
        "--shape", type=_parse_shape,
        help="optional arbitrary shape such as 1x197x768; overrides --batch/--count",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--distribution", choices=("normal", "uniform"),
                        default="normal")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--mu", type=float, default=MU0)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    args = parser.parse_args()
    if torch.cuda.device_count() < 3:
        parser.error("at least three CUDA devices are required")
    if args.batch < 1 or args.count < 1 or args.repeats < 1:
        parser.error("require batch>=1, count>=1 and repeats>=1")
    args.shape = args.shape or (args.batch, args.count)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cryptgpu-gelu-",
                                     dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), options),
                 nprocs=3, join=True)


if __name__ == "__main__":
    main()
