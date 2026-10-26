"""Compare sGeLU with CrypTen's native MPC-erf GeLU on the same 3PC input."""

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import sys

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
from .bridge import ExperimentalG2Bridge
from .sgelu import MU0, error_metrics, g2
from .runtime import Ring3, Ring3Wide

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "tmp" / "gelu-baseline"


def start_parties(worker, rendezvous, options):
    """Start CPU/Gloo parties without reinitializing macOS OpenMP in children."""
    method = "fork" if sys.platform == "darwin" else "spawn"
    if sys.platform == "darwin":
        # Gloo's macOS libuv backend cannot resolve the default hostname in
        # this environment; en0 is the active local interface here.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "en0")
    mp.start_processes(worker, args=(rendezvous, options), nprocs=3,
                       join=True, start_method=method)


def _stats():
    import crypten
    stats = crypten.communicator.get().get_communication_stats()
    return torch.tensor([stats["rounds"], stats["bytes"], stats["time"]], dtype=torch.float64)


def _gather_metrics(metrics):
    gathered = [torch.empty_like(metrics) for _ in range(3)]
    dist.all_gather(gathered, metrics)
    return torch.stack(gathered)


def worker(rank, rendezvous, options):
    import crypten

    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    crypten.init()
    crypten.cfg.communicator.verbose = True
    try:
        base = Ring3()
        wide = Ring3Wide(base)
        batch, count, mu = options["batch"], options["count"], options["mu"]
        generator = torch.Generator().manual_seed(options["seed"])
        if options["distribution"] == "linspace":
            x = torch.linspace(-8, 8, batch * count, dtype=torch.float64)
        elif options["distribution"] == "normal":
            x = (1.5 * torch.randn(batch * count, generator=generator,
                                    dtype=torch.float64)).clamp(-8, 8)
        else:
            x = torch.empty(batch * count, dtype=torch.float64).uniform_(-3, 3,
                                                                           generator=generator)
        x = x.reshape(batch, count)
        x_q17 = (x * (1 << 17)).round().long()
        quantized_x = x_q17.double() / (1 << 17)
        exact = F.gelu(x, approximate="none")
        mask_generator = torch.Generator().manual_seed(options["seed"] + 1)
        mask = (2.0 * torch.pi * torch.rand(x.shape, generator=mask_generator,
                                             dtype=torch.float64) - torch.pi)
        harmonics = torch.arange(1, 8, dtype=torch.float64).reshape(1, 1, 7)
        mask_sin_public = torch.sin(mask.unsqueeze(-1) * harmonics * torch.pi / 10.0)
        mask_cos_public = torch.cos(mask.unsqueeze(-1) * harmonics * torch.pi / 10.0)
        records = []

        for iteration in range(options["repeats"] + 1):
            dist.barrier()
            started = time.perf_counter()
            encrypted = crypten.cryptensor(quantized_x, src=0, precision=17)
            comparison_encrypted = crypten.cryptensor(x_q17, src=0, precision=0)
            mask_encrypted = crypten.cryptensor(mask, src=0, precision=17)
            mask_sin = crypten.cryptensor(mask_sin_public, src=0, precision=17)
            mask_cos = crypten.cryptensor(mask_cos_public, src=0, precision=17)
            input_seconds = time.perf_counter() - started

            dist.barrier()
            bridge = ExperimentalG2Bridge(base, wide, mu)
            started = time.perf_counter()
            bridge.prepare(encrypted.share.shape)
            sgelu_offline_seconds = time.perf_counter() - started
            dist.barrier()
            sgelu_bytes_before, sgelu_phases_before = base.sent_bytes, base.phases
            started = time.perf_counter()
            sgelu_encrypted = bridge(encrypted)
            sgelu_online_seconds = time.perf_counter() - started
            sgelu_bytes = base.sent_bytes - sgelu_bytes_before
            sgelu_phases = base.phases - sgelu_phases_before
            sgelu_plain = sgelu_encrypted.get_plain_text()

            # CrypTen-native PUMA polynomial baseline.  Unlike sGeLU, this
            # baseline remains entirely inside CrypTen and uses its LT masks.
            dist.barrier()
            crypten.reset_communication_stats()
            started = time.perf_counter()
            puma_encrypted = crypten_puma_gelu(
                encrypted, comparison_value=comparison_encrypted
            )
            puma_online_seconds = time.perf_counter() - started
            puma_stats = _stats()
            puma_plain = puma_encrypted.get_plain_text()

            dist.barrier()
            crypten.reset_communication_stats()
            started = time.perf_counter()
            baseline_encrypted = crypten_erf_gelu(encrypted)
            baseline_online_seconds = time.perf_counter() - started
            baseline_stats = _stats()
            baseline_plain = baseline_encrypted.get_plain_text()

            # MPCFormer Quad baseline: one secret square plus public affine terms.
            dist.barrier()
            crypten.reset_communication_stats()
            started = time.perf_counter()
            quad_encrypted = crypten_quad_gelu(encrypted)
            quad_online_seconds = time.perf_counter() - started
            quad_stats = _stats()
            quad_plain = quad_encrypted.get_plain_text()

            dist.barrier()
            crypten.reset_communication_stats()
            started = time.perf_counter()
            secformer_encrypted = crypten_secformer_fourier_gelu(
                encrypted, comparison_value=comparison_encrypted
            )
            secformer_online_seconds = time.perf_counter() - started
            secformer_stats = _stats()
            secformer_plain = secformer_encrypted.get_plain_text()

            dist.barrier()
            crypten.reset_communication_stats()
            started = time.perf_counter()
            masked_encrypted = crypten_secformer_masked_fourier_gelu(
                encrypted, mask_encrypted, mask_sin, mask_cos,
                comparison_value=comparison_encrypted
            )
            masked_online_seconds = time.perf_counter() - started
            masked_stats = _stats()
            masked_plain = masked_encrypted.get_plain_text()


            sgelu_metrics = torch.tensor(
                [input_seconds, sgelu_offline_seconds, sgelu_online_seconds,
                 sgelu_bytes, sgelu_phases], dtype=torch.float64)
            baseline_metrics = torch.tensor(
                [baseline_online_seconds, baseline_stats[0], baseline_stats[1],
                 baseline_stats[2]], dtype=torch.float64)
            quad_metrics = torch.tensor(
                [quad_online_seconds, quad_stats[0], quad_stats[1],
                 quad_stats[2]], dtype=torch.float64)
            secformer_metrics = torch.tensor(
                [secformer_online_seconds, secformer_stats[0],
                 secformer_stats[1], secformer_stats[2]], dtype=torch.float64)
            masked_metrics = torch.tensor(
                [masked_online_seconds, masked_stats[0], masked_stats[1],
                 masked_stats[2]], dtype=torch.float64)
            puma_metrics = torch.tensor(
                [puma_online_seconds, puma_stats[0], puma_stats[1], puma_stats[2]],
                dtype=torch.float64)
            sgelu_all = _gather_metrics(sgelu_metrics)
            baseline_all = _gather_metrics(baseline_metrics)
            quad_all = _gather_metrics(quad_metrics)
            secformer_all = _gather_metrics(secformer_metrics)
            masked_all = _gather_metrics(masked_metrics)
            puma_all = _gather_metrics(puma_metrics)

            if rank == 0:
                sgelu_result = sgelu_plain.double()
                baseline_result = baseline_plain.double()
                quad_result = quad_plain.double()
                secformer_result = secformer_plain.double()
                masked_result = masked_plain.double()
                records.append({
                    "warmup": iteration == 0,
                    "input_sharing_seconds_max": sgelu_all[:, 0].max().item(),
                    "sgelu_offline_seconds_max": sgelu_all[:, 1].max().item(),
                    "sgelu_online_seconds_max": sgelu_all[:, 2].max().item(),
                    "sgelu_sent_bytes_sum": int(sgelu_all[:, 3].sum()),
                    "sgelu_exchange_phases_per_rank": sgelu_all[:, 4].int().tolist(),
                    "crypten_baseline_online_seconds_max": baseline_all[:, 0].max().item(),
                    "crypten_baseline_rounds_per_rank": baseline_all[:, 1].int().tolist(),
                    "crypten_baseline_bytes_sum": int(baseline_all[:, 2].sum()),
                    "crypten_baseline_comm_time_sum": baseline_all[:, 3].sum().item(),
                    "mpcformer_quad_online_seconds_max": quad_all[:, 0].max().item(),
                    "mpcformer_quad_rounds_per_rank": quad_all[:, 1].int().tolist(),
                    "mpcformer_quad_bytes_sum": int(quad_all[:, 2].sum()),
                    "mpcformer_quad_comm_time_sum": quad_all[:, 3].sum().item(),
                    "secformer_fourier_online_seconds_max": secformer_all[:, 0].max().item(),
                    "secformer_fourier_rounds_per_rank": secformer_all[:, 1].int().tolist(),
                    "secformer_fourier_bytes_sum": int(secformer_all[:, 2].sum()),
                    "secformer_fourier_comm_time_sum": secformer_all[:, 3].sum().item(),
                    "secformer_masked_fourier_online_seconds_max": masked_all[:, 0].max().item(),
                    "secformer_masked_fourier_rounds_per_rank": masked_all[:, 1].int().tolist(),
                    "secformer_masked_fourier_bytes_sum": int(masked_all[:, 2].sum()),
                    "secformer_masked_fourier_comm_time_sum": masked_all[:, 3].sum().item(),
                    "sgelu_vs_float_g2": error_metrics(sgelu_result, g2(quantized_x, mu)),
                    "crypten_baseline_vs_exact_gelu": error_metrics(baseline_result, exact),
                    "mpcformer_quad_vs_exact_gelu": error_metrics(quad_result, exact),
                    "secformer_fourier_vs_exact_gelu": error_metrics(secformer_result, exact),
                    "secformer_masked_fourier_vs_exact_gelu": error_metrics(masked_result, exact),
                    "secformer_fourier_reference_vs_exact_gelu": error_metrics(
                        secformer_fourier_gelu_reference(quantized_x), exact
                    ),
                    "sgelu_vs_exact_gelu": error_metrics(sgelu_result, exact),
                    "baseline_vs_sgelu": error_metrics(baseline_result, sgelu_result),
                    "puma_crypten_online_seconds_max": puma_all[:, 0].max().item(),
                    "puma_crypten_rounds_per_rank": puma_all[:, 1].int().tolist(),
                    "puma_crypten_bytes_sum": int(puma_all[:, 2].sum()),
                    "puma_crypten_comm_time_sum": puma_all[:, 3].sum().item(),
                    "puma_crypten_vs_puma_reference": error_metrics(
                        puma_plain.double(), puma_gelu_reference(x)),
                    "puma_crypten_vs_exact_gelu": error_metrics(puma_plain.double(), exact),
                })

        if rank == 0:
            output = Path(options["output"])
            report = {
                "status": "benchmark_passed",
                "fixture": f"synthetic {options['distribution']}; NOT real model activations",
                "python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "world_size": 3, "backend": "gloo",
                "shape": [batch, count], "mu": mu, "q_input": 17,
                "seed": options["seed"], "distribution": options["distribution"],
                "baseline": "CrypTen erf Taylor approximation",
                "mpcformer_quad": "0.125*x^2 + 0.25*x + 0.5",
                "secformer_fourier": "direct CrypTen sine; precision-zero integer copy for comparisons",
                "secformer_masked_fourier": "masked opening reference; not a standalone security claim",
                "puma": "PUMA published piecewise polynomial evaluated natively with CrypTen arithmetic and encrypted LT masks",
                "baseline_erf_iterations": int(crypten.cfg.functions.erf_iterations),
                "communication_note": "CrypTen bytes are summed across parties; its documented overall-system estimate divides this by 2. sGeLU reports custom-ring sent bytes summed across parties.",
                "records": records,
            }
            output.mkdir(parents=True, exist_ok=True)
            (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(output / "comparison.json"),
                              "sgelu_online_seconds": records[-1]["sgelu_online_seconds_max"],
                              "baseline_online_seconds": records[-1]["crypten_baseline_online_seconds_max"],
                              "mpcformer_quad_online_seconds": records[-1]["mpcformer_quad_online_seconds_max"],
                              "secformer_fourier_online_seconds": records[-1]["secformer_fourier_online_seconds_max"],
                              "sgelu_bytes": records[-1]["sgelu_sent_bytes_sum"],
                              "baseline_bytes": records[-1]["crypten_baseline_bytes_sum"],
                              "mpcformer_quad_bytes": records[-1]["mpcformer_quad_bytes_sum"],
                              "secformer_fourier_bytes": records[-1]["secformer_fourier_bytes_sum"],
                              "secformer_masked_fourier_online_seconds": records[-1]["secformer_masked_fourier_online_seconds_max"],
                              "puma_crypten_online_seconds": records[-1]["puma_crypten_online_seconds_max"],
                              "secformer_masked_fourier_bytes": records[-1]["secformer_masked_fourier_bytes_sum"]}, indent=2), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=257)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mu", type=float, default=MU0)
    parser.add_argument("--distribution", choices=["linspace", "normal", "uniform"], default="linspace")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.count < 1 or args.batch < 1 or args.repeats < 1:
        parser.error("require batch>=1, count>=1 and repeats>=1")
    args.output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="gelu-benchmark-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        start_parties(worker, Path(directory, "store").as_uri(), options)


if __name__ == "__main__":
    main()
