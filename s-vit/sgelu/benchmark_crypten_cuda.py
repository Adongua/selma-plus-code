"""CrypTen-native three-party CUDA/NCCL GeLU microbenchmark.

The baselines use CrypTen's CUDA MPCTensor implementation.  The manuscript
sGeLU protocol is wrapped around the same CrypTen process group because its
96-bit Ruler arithmetic is not an upstream CrypTen primitive.
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

from .baseline import (
    crypten_erf_gelu,
    crypten_puma_gelu,
    crypten_quad_gelu,
    crypten_secformer_fourier_gelu,
    crypten_secformer_masked_fourier_gelu,
    puma_gelu_reference,
    secformer_fourier_gelu_reference,
)
from .cuda_baseline import prepare_fourier, prepare_erf, prepare_quad
from .protocol import prepare_g2, ruler_g2
from .runtime_cuda import CudaRing3, CudaRing3Wide
from .sgelu import MU0, error_metrics, g2

METHODS = ("sgelu", "puma", "quad", "erf", "fourier", "masked_fourier")


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 1x197x768") from error
    if not shape or any(dimension < 1 for dimension in shape):
        raise argparse.ArgumentTypeError("all dimensions must be positive")
    return shape


def _sync(device):
    torch.cuda.synchronize(device)


def _aggregate(device, seconds, stats):
    row = torch.tensor(
        [seconds, stats["bytes"], stats["rounds"], stats["time"]],
        dtype=torch.float64,
        device=device,
    )
    rows = [torch.empty_like(row) for _ in range(3)]
    dist.all_gather(rows, row)
    values = torch.stack(rows).cpu()
    return {
        "seconds_max": values[:, 0].max().item(),
        "sent_bytes_sum": int(values[:, 1].sum().item()),
        "rounds_per_rank": values[:, 2].int().tolist(),
        "comm_time_sum": values[:, 3].sum().item(),
    }


def _crypten_measure(device, crypten, function):
    dist.barrier()
    crypten.reset_communication_stats()
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    stats = crypten.communicator.get().get_communication_stats()
    seconds = time.perf_counter() - started
    return result, _aggregate(device, seconds, stats)


def _ring_measure(device, base, function):
    dist.barrier()
    _sync(device)
    bytes_before, rounds_before = base.sent_bytes, base.phases
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    metrics = {
        "bytes": base.sent_bytes - bytes_before,
        "rounds": base.phases - rounds_before,
        "time": 0.0,
    }
    return result, _aggregate(device, seconds, metrics)


def _plain(value):
    return (value.tensor() if hasattr(value, "tensor") else value).double()


class CrypTenCudaG2:
    """sGeLU adapter using CrypTen's initialized CUDA process group."""

    def __init__(self, device, mu):
        from crypten.cuda import CUDALongTensor
        from crypten.mpc import MPCTensor, ptype

        self.CUDALongTensor = CUDALongTensor
        self.MPCTensor = MPCTensor
        self.ptype = ptype
        self.base = CudaRing3(device=device, reveal_mode="p2p")
        self.wide = CudaRing3Wide(self.base)
        self.mu = mu
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("pending sGeLU material must be consumed first")
        self.material = prepare_g2(self.base, self.wide, tuple(shape), self.mu)

    def __call__(self, value):
        if value.ptype != self.ptype.arithmetic:
            raise ValueError("sGeLU requires arithmetic CrypTen shares")
        if value.encoder._precision_bits != 17:
            raise ValueError("sGeLU requires Q17 input")
        if self.material is None:
            raise RuntimeError("prepare sGeLU material first")
        material, self.material = self.material, None
        output = ruler_g2(self.base, self.wide, value.share.data, material)
        return self.MPCTensor.from_shares(
            self.CUDALongTensor(output), precision=17
        )


def _method_reference(x, method, mu):
    if method == "sgelu":
        return g2(x, mu)
    if method == "puma":
        return puma_gelu_reference(x)
    if method == "quad":
        return 0.125 * x.square() + 0.25 * x + 0.5
    if method in ("erf", "fourier", "masked_fourier"):
        return (F.gelu(x, approximate="none") if method == "erf"
                else secformer_fourier_gelu_reference(x))
    raise ValueError(method)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(
        RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
        DISTRIBUTED_BACKEND="nccl",
    )
    import crypten

    crypten.init(device=device)
    crypten.cfg.communicator.verbose = False
    try:
        generator = torch.Generator().manual_seed(options["seed"])
        shape = tuple(options["shape"])
        x_cpu = (1.5 * torch.randn(shape, generator=generator, dtype=torch.float64)).clamp(-8, 8)
        x = x_cpu.to(device)
        quantized = ((x_cpu * (1 << 17)).round() / (1 << 17)).to(device)
        comparison = (x_cpu * (1 << 17)).round().long().to(device)
        exact = F.gelu(x, approximate="none")
        references = {
            method: _method_reference(x, method, options["mu"])
            for method in options["methods"]
        }

        mask_generator = torch.Generator().manual_seed(options["seed"] + 1)
        mask_cpu = 2.0 * torch.pi * torch.rand(shape, generator=mask_generator, dtype=torch.float64) - torch.pi
        harmonics = torch.arange(1, 8, dtype=torch.float64).reshape((1,) * len(shape) + (7,))
        mask_sin_cpu = torch.sin(mask_cpu.unsqueeze(-1) * harmonics * torch.pi / 10.0)
        mask_cos_cpu = torch.cos(mask_cpu.unsqueeze(-1) * harmonics * torch.pi / 10.0)
        records = []

        if rank == 0 and options["progress"]:
            print(json.dumps({
                "event": "initialized",
                "framework": "CrypTen MPCTensor / NCCL / CUDA",
                "shape": list(shape),
                "methods": options["methods"],
            }), flush=True)

        for iteration in range(options["repeats"] + 1):
            encrypted, input_stats = _crypten_measure(
                device, crypten,
                lambda: crypten.cryptensor(quantized, src=0, precision=17, device=device),
            )
            comparison_encrypted = crypten.cryptensor(
                comparison, src=0, precision=0, device=device
            )
            mask = crypten.cryptensor(mask_cpu.to(device), src=0, precision=17, device=device)
            mask_sin = crypten.cryptensor(mask_sin_cpu.to(device), src=0, precision=17, device=device)
            mask_cos = crypten.cryptensor(mask_cos_cpu.to(device), src=0, precision=17, device=device)
            row = {"warmup": iteration == 0, "input_sharing": input_stats, "methods": {}}

            for method in options["methods"]:
                if method == "sgelu":
                    operator = CrypTenCudaG2(device, options["mu"])
                    _, offline = _ring_measure(
                        device, operator.base,
                        lambda operator=operator: operator.prepare(encrypted.share.shape),
                    )
                    output, online = _ring_measure(
                        device, operator.base,
                        lambda operator=operator: operator(encrypted.clone()),
                    )
                else:
                    if method == "puma":
                        function = lambda: crypten_puma_gelu(
                            encrypted.clone(), comparison_value=comparison_encrypted
                        )
                    elif method == "quad":
                        function = lambda: crypten_quad_gelu(encrypted.clone())
                    elif method == "erf":
                        function = lambda: crypten_erf_gelu(encrypted.clone())
                    elif method == "fourier":
                        function = lambda: crypten_secformer_fourier_gelu(
                            encrypted.clone(), comparison_value=comparison_encrypted
                        )
                    else:
                        function = lambda: crypten_secformer_masked_fourier_gelu(
                            encrypted.clone(), mask, mask_sin, mask_cos,
                            comparison_value=comparison_encrypted,
                        )
                    output, offline = _crypten_measure(device, crypten, function)
                    online = offline
                opened, reconstruction = _crypten_measure(
                    device, crypten, lambda output=output: output.get_plain_text()
                )
                if rank == 0:
                    actual = _plain(opened)
                    row["methods"][method] = {
                        "offline": offline,
                        "online": online,
                        "output_reconstruction": reconstruction,
                        "error_vs_method_reference": error_metrics(actual, references[method]),
                        "error_vs_torch_gelu": error_metrics(actual, exact),
                    }
                    if options["progress"]:
                        print(json.dumps({
                            "event": "method_complete",
                            "warmup": iteration == 0,
                            "method": method,
                            "online_seconds": online["seconds_max"],
                            "online_bytes": online["sent_bytes_sum"],
                        }), flush=True)
            if rank == 0:
                records.append(row)

        if rank == 0:
            output = Path(options["output"])
            output.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "framework": "CrypTen-main MPCTensor / NCCL / CUDA",
                "world_size": 3,
                "shape": list(shape),
                "methods": options["methods"],
                "repeats": options["repeats"],
                "mu": options["mu"],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fixture": "synthetic normal generated by CPU RNG before timing",
                "method_provenance": {
                    "sgelu": "manuscript protocol using the Ring3 adapter on the CrypTen NCCL process group",
                    "puma": "PUMA piecewise polynomial re-evaluated with CrypTen MPCTensor",
                    "quad": "quadratic approximation evaluated with CrypTen MPCTensor",
                    "erf": "CrypTen MPCTensor erf Taylor approximation",
                    "fourier": "SecFormer Fourier coefficients evaluated with CrypTen MPCTensor",
                    "masked_fourier": "masked Fourier engineering reference; not a standalone security claim",
                },
                "security": "CrypTen GPU benchmark; sGeLU Ruler privacy caveats remain",
                "records": records,
            }
            path = output / "comparison.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(path), "records": len(records)}, indent=2), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", type=_parse_shape, default=(1, 17))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--mu", type=float, default=MU0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="disable rank-0 phase progress messages")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 3:
        parser.error("at least three CUDA devices are required")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crypten-gelu-cuda-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["shape"] = tuple(args.shape)
        options["output"] = str(args.output)
        mp.spawn(worker, args=(Path(directory, "store").as_uri(), options), nprocs=3, join=True)


if __name__ == "__main__":
    main()
