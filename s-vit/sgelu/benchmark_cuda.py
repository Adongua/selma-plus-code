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
from .cuda_baseline import (prepare_erf, prepare_fourier, prepare_quad,
                             erf_gelu, fourier_gelu, quad_gelu)
from .baseline import puma_gelu_reference, secformer_fourier_gelu_reference
from .puma_baseline import prepare_puma, puma_gelu
from .runtime_cuda import CudaRing3, CudaRing3Wide
from .sgelu import MU0, error_metrics, g2


PROFILE_STAGES = ("square_and_joint_trunc", "scale_extension",
                  "wide_mul_reveal_isqrt", "inverse_trunc",
                  "bracket_trunc", "product_and_trunc")


class OnlineProfiler:
    """Diagnostic-only stage timer; synchronization changes the measured path."""

    def __init__(self, device):
        self.device = device
        self.started = {}
        self.seconds = {}

    def start(self, name):
        torch.cuda.synchronize(self.device)
        self.started[name] = time.perf_counter()

    def stop(self, name):
        torch.cuda.synchronize(self.device)
        self.seconds[name] = time.perf_counter() - self.started.pop(name)


def gather_profile(base, profiler):
    if profiler is None:
        return None
    values = torch.tensor([profiler.seconds.get(name, 0.0) for name in PROFILE_STAGES],
                          dtype=torch.float64, device=base.device)
    gathered = [torch.empty_like(values) for _ in range(3)]
    dist.all_gather(gathered, values)
    rows = torch.stack(gathered).cpu()
    return {name: rows[:, index].max().item() for index, name in enumerate(PROFILE_STAGES)}


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
        base, wide = CudaRing3(device=device, reveal_mode=options["reveal_mode"]), None
        wide = CudaRing3Wide(base)
        # Generate the public synthetic fixture with the CPU RNG and then copy
        # it before timing.  This exactly matches benchmark.py, allowing a
        # sample-for-sample CPU/CUDA correctness comparison while keeping the
        # secure operator hot path entirely on CUDA.
        generator = torch.Generator().manual_seed(options["seed"])
        x = (1.5 * torch.randn((options["batch"], options["count"]),
                               generator=generator, dtype=torch.float64)
             .clamp(-8, 8).to(device))
        x_q17 = (x * (1 << 17)).round().long()
        target = F.gelu(x, approximate="none")
        target_g2 = g2(x_q17.double() / (1 << 17), options["mu"])
        records = []
        for iteration in range(options["repeats"] + 1):
            shares, input_metrics = measure(base, lambda: base.private(x_q17, 0))
            material, offline = measure(
                base, lambda: prepare_g2(base, wide, x_q17.shape, options["mu"]))
            profiler = OnlineProfiler(base.device) if options["profile_online"] else None
            output, online = measure(
                base, lambda: ruler_g2(base, wide, shares, material, profiler=profiler))
            profile = gather_profile(base, profiler)
            opened, reconstruction = measure(base, lambda: base.reveal(output))
            quad_material, quad_offline = measure(
                base, lambda: prepare_quad(base, x_q17.shape))
            quad_output, quad_online = measure(
                base, lambda: quad_gelu(base, shares, quad_material))
            quad_opened, quad_reconstruction = measure(
                base, lambda: base.reveal(quad_output))
            erf_material, erf_offline = measure(
                base, lambda: prepare_erf(base, x_q17.shape))
            erf_output, erf_online = measure(
                base, lambda: erf_gelu(base, shares, erf_material))
            erf_opened, erf_reconstruction = measure(
                base, lambda: base.reveal(erf_output))
            fourier_material, fourier_offline = measure(
                base, lambda: prepare_fourier(base, x_q17.shape))
            fourier_output, fourier_online = measure(
                base, lambda: fourier_gelu(base, shares, fourier_material))
            fourier_opened, fourier_reconstruction = measure(
                base, lambda: base.reveal(fourier_output))
            puma_material, puma_offline = measure(
                base, lambda: prepare_puma(base, x_q17.shape))
            puma_output, puma_online = measure(
                base, lambda: puma_gelu(base, shares, puma_material))
            puma_opened, puma_reconstruction = measure(
                base, lambda: base.reveal(puma_output))
            if rank == 0:
                result = opened.double() / (1 << 17)
                quad_result = quad_opened.double() / (1 << 17)
                erf_result = erf_opened.double() / (1 << 17)
                fourier_result = fourier_opened.double() / (1 << 17)
                puma_result = puma_opened.double() / (1 << 17)
                error = error_metrics(result, target)
                records.append({"warmup": iteration == 0,
                                "input_sharing": input_metrics, "offline": offline,
                                "online": online, "output_reconstruction": reconstruction,
                                "online_profile_seconds_max": profile,
                                "quad_offline": quad_offline, "quad_online": quad_online,
                                "quad_output_reconstruction": quad_reconstruction,
                                "crypten_erf_offline": erf_offline,
                                "crypten_erf_online": erf_online,
                                "crypten_erf_output_reconstruction": erf_reconstruction,
                                "secformer_fourier_offline": fourier_offline,
                                "secformer_fourier_online": fourier_online,
                                "secformer_fourier_output_reconstruction": fourier_reconstruction,
                                "puma_style_offline": puma_offline,
                                "puma_style_online": puma_online,
                                "puma_style_output_reconstruction": puma_reconstruction,
                                "error_vs_float_g2": error_metrics(result, target_g2),
                                "error_vs_torch_gelu": error,
                                "quad_vs_torch_gelu": error_metrics(quad_result, target),
                                "crypten_erf_vs_torch_gelu": error_metrics(erf_result, target),
                                "secformer_fourier_vs_torch_gelu": error_metrics(fourier_result, target),
                                "secformer_fourier_reference_vs_torch_gelu": error_metrics(
                                    secformer_fourier_gelu_reference(x.cpu()).to(base.device), target)})
                records[-1]["puma_style_vs_puma_reference"] = error_metrics(
                    puma_result, puma_gelu_reference(x))
                records[-1]["puma_style_vs_torch_gelu"] = error_metrics(puma_result, target)
        if rank == 0:
            output = Path(options["output"])
            output.mkdir(parents=True, exist_ok=True)
            report = {"status": "executed_not_security_validated", "world_size": 3,
                      "backend": "NCCL / CUDA", "shape": [options["batch"], options["count"]],
                      "mu": options["mu"], "repeats": options["repeats"],
                      "reveal_mode": options["reveal_mode"],
                      "profile_online": options["profile_online"],
                      "fixture": "synthetic normal generated by CPU RNG before timing; matches CPU benchmark",
                      "python": platform.python_version(), "torch": torch.__version__,
                      "cuda_hot_path": {"public_isqrt": "CUDA float64 seed plus exact integer correction",
                                        "random_expansion": "CUDA Philox",
                                        "host_tensor_roundtrip": False},
                      "cuda_quad": "same CUDA Ring3 fixed-point port of 0.125*x^2 + 0.25*x + 0.5",
                      "cuda_crypten_erf": "Q17 fixed-point port of CrypTen erf Taylor approximation with 8 iterations",
                      "cuda_secformer_fourier": "CUDA masked-Fourier port with precomputed secret mask sine/cosine and local CUDA sine/cosine on opened delta; interval threshold is opened publicly because CUDA comparison is not implemented",
                      "cuda_puma_style": "PUMA published piecewise cubic approximation ported to CUDA Ring3; interval selection opens Q17 x because secure less-than is not implemented",
                      "records": records,
                      "security": "Ruler remains not security validated; CUDA Philox is a benchmark PRG, not a reviewed CSPRNG"}
            (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            last = records[-1]
            print(json.dumps({"report": str(output / "comparison.json"),
                              "records": len(records),
                              "sgelu_online_seconds": last["online"]["seconds_max"],
                              "quad_online_seconds": last["quad_online"]["seconds_max"],
                              "crypten_erf_online_seconds": last["crypten_erf_online"]["seconds_max"],
                              "secformer_fourier_online_seconds": last["secformer_fourier_online"]["seconds_max"],
                              "puma_style_online_seconds": last["puma_style_online"]["seconds_max"],
                              "sgelu_bytes": last["online"]["sent_bytes_sum"],
                              "quad_bytes": last["quad_online"]["sent_bytes_sum"],
                              "crypten_erf_bytes": last["crypten_erf_online"]["sent_bytes_sum"],
                              "secformer_fourier_bytes": last["secformer_fourier_online"]["sent_bytes_sum"],
                              "puma_style_bytes": last["puma_style_online"]["sent_bytes_sum"],
                              "puma_style_mae_vs_exact_gelu": last["puma_style_vs_torch_gelu"]["mae"]},
                             indent=2), flush=True)
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
    parser.add_argument("--reveal-mode", choices=("all_reduce", "all_gather", "p2p"),
                        default="p2p",
                        help="one-phase NCCL collective or legacy two-phase ring opening")
    parser.add_argument("--profile-online", action="store_true",
                        help="record synchronized per-stage timings; diagnostic only")
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
