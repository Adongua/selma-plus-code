"""Synthetic three-party functional verification, with separate timing phases.

Run via `python -m sgelu.validate`. No datasets/checkpoints are downloaded.
All opened test inputs and reference outputs here are synthetic/public fixtures.
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

from .sgelu import MU0, g2, error_metrics
from .protocol import (prepare_g2, ruler_g2, extension_material, extend_online,
                       prob_trunc_material, prob_trunc)
from .runtime import Ring3, Ring3Wide
from .wide import limbs_from_uint64, mul96

DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "tmp" / "gelu-v2"


def assert_equal(actual, expected):
    if not torch.equal(actual, expected):
        raise AssertionError("integer protocol result differs from its reference")


def worker(rank, rendezvous, options):
    torch.set_num_threads(1)
    # File rendezvous is placed under the project tmp, and Gloo uses loopback.
    if options["crypten"]:
        import crypten
        os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                          DISTRIBUTED_BACKEND="gloo")
        crypten.init()
    else:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                                timeout=timedelta(seconds=180))
    try:
        base = Ring3()
        wide = Ring3Wide(base)
        primitives(base, wide, options["primitive_count"])
        count = options["count"]
        x = torch.linspace(-8, 8, count, dtype=torch.float64).reshape(1, count)
        x_q17 = (x * (1 << 17)).round().long()
        quantized_x = x_q17.double() / (1 << 17)
        mu = options["mu"]
        records = []
        for iteration in range(options["repeats"] + 1):
            dist.barrier()
            started = time.perf_counter()
            share = base.private(x_q17 if rank == 0 else torch.zeros_like(x_q17), 0)
            input_seconds = time.perf_counter() - started
            prepare_bytes, prepare_phases = base.sent_bytes, base.phases
            started = time.perf_counter()
            material = prepare_g2(base, wide, tuple(share.shape), mu)
            prepare_seconds = time.perf_counter() - started
            prepare_bytes, prepare_phases = base.sent_bytes - prepare_bytes, base.phases - prepare_phases
            dist.barrier()  # excluded from online interval
            online_bytes, online_phases = base.sent_bytes, base.phases
            started = time.perf_counter()
            result_share = ruler_g2(base, wide, share, material)
            online_seconds = time.perf_counter() - started
            online_bytes, online_phases = base.sent_bytes - online_bytes, base.phases - online_phases
            started = time.perf_counter()
            result = base.reveal(result_share).double() / (1 << 17)
            opening_seconds = time.perf_counter() - started
            if not torch.isfinite(result).all():
                raise AssertionError("nonfinite G2 output")
            protocol_error = error_metrics(result, g2(quantized_x, mu))
            if protocol_error["max_abs"] >= .005:
                raise AssertionError(f"protocol error exceeds 0.005: {protocol_error}")
            try:
                ruler_g2(base, wide, share, material)
            except RuntimeError:
                pass
            else:
                raise AssertionError("material was reused")
            metrics = torch.tensor([input_seconds, prepare_seconds, online_seconds, opening_seconds,
                                    prepare_bytes, online_bytes, prepare_phases, online_phases], dtype=torch.float64)
            gathered = [torch.empty_like(metrics) for _ in range(3)]
            dist.all_gather(gathered, metrics)  # reporting traffic not charged to operator
            if rank == 0:
                stacked = torch.stack(gathered)
                records.append({"warmup": iteration == 0,
                    "input_sharing_seconds_max": stacked[:,0].max().item(),
                    "prepare_seconds_max": stacked[:,1].max().item(),
                    "online_seconds_max": stacked[:,2].max().item(),
                    "output_opening_seconds_max": stacked[:,3].max().item(),
                    "prepare_sent_bytes_sum": int(stacked[:,4].sum()),
                    "online_sent_bytes_sum": int(stacked[:,5].sum()),
                    "prepare_exchange_phases_per_rank": stacked[:,6].int().tolist(),
                    "online_exchange_phases_per_rank": stacked[:,7].int().tolist(),
                    "protocol_vs_float_g2_on_quantized_input": protocol_error,
                    "float_g2_vs_exact_gelu": error_metrics(g2(x, mu), F.gelu(x, approximate="none")),
                    "total_vs_exact_gelu": error_metrics(result, F.gelu(x, approximate="none"))})
        # End-to-end adapter test is optional because the old local CrypTen has
        # additional dependency constraints. It is not part of timing samples.
        bridge_status = "not_requested"
        if options["crypten"]:
            from .bridge import ExperimentalG2Bridge
            encrypted = crypten.cryptensor(quantized_x, src=0, precision=17)
            bridge = ExperimentalG2Bridge(base, wide, mu)
            bridge.prepare(share.shape)
            try:
                bridge.prepare(share.shape)
            except RuntimeError:
                pass
            else:
                raise AssertionError("bridge overwrote pending one-shot material")
            result = bridge(encrypted)
            actual = result.get_plain_text().double()
            if error_metrics(actual, g2(quantized_x, mu))["max_abs"] >= .005:
                raise AssertionError("CrypTen bridge mismatch")
            if result.encoder._precision_bits != 17:
                raise AssertionError("CrypTen bridge changed precision")
            try:
                bridge(encrypted)
            except RuntimeError:
                pass
            else:
                raise AssertionError("bridge accepted missing material")
            bridge_status = "passed_crypten_encrypt_forward_decrypt"
        if rank == 0:
            report = {"status": "functional_checks_passed", "privacy_status": "not_secure_as_specified",
                      "fixture": "synthetic linspace [-8,8]; NOT real model activations",
                      "python": platform.python_version(), "torch": torch.__version__,
                      "platform": platform.platform(), "world_size": 3, "backend": "gloo",
                      "threads_per_party": 1, "shape": list(x.shape), "mu": mu,
                      "q_input_output": 17, "independent_branch_scales": True,
                      "primitive_count": options["primitive_count"], "crypten_bridge": bridge_status,
                      "records": records}
            output = Path(options["output"])
            (output / "three_party.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"report": str(output / "three_party.json"),
                              "worst_protocol_error": max(r["protocol_vs_float_g2_on_quantized_input"]["max_abs"] for r in records),
                              "crypten_bridge": bridge_status}, indent=2), flush=True)
    finally:
        if options["crypten"]:
            crypten.uninit()
        else:
            dist.destroy_process_group()


def primitives(base, wide, count):
    # Includes all input owners and full 64-bit wraparound, not just tiny shares.
    generator = torch.Generator().manual_seed(20261024)
    x = torch.randint(-(1 << 62), 1 << 62, (count,), generator=generator)
    y = torch.randint(-(1 << 62), 1 << 62, (count,), generator=generator)
    for owner in range(3):
        sx = base.private(x if base.rank == owner else torch.zeros_like(x), owner)
        sy = base.private(y if base.rank == owner else torch.zeros_like(y), owner)
        assert_equal(base.reveal(base.refresh(sx)), x)
        assert_equal(base.reveal(base.mul(sx, sy)), x * y)
        assert_equal(base.reveal(base.square(sx)), x.square())
    wx, wy = limbs_from_uint64(x), limbs_from_uint64(y)
    sx, sy = wide.private(wx, 0), wide.private(wy, 1)
    assert_equal(wide.reveal(wide.mul(sx, sy)), mul96(wx, wy))
    assert_equal(wide.reveal(wide.refresh(sx)), wx)
    values = torch.randint(0, 1 << 60, (count,), generator=generator)
    values[:4] = torch.tensor([0, 1, (1 << 60)-1, (1 << 62)-1])
    share = base.private(values, 2)
    mask = extension_material(base, wide, values.shape)
    # Audit-only openings: these synthetic mask fixtures are NEVER used online.
    mask64 = base.reveal(mask[0])
    assert bool(((mask64 >= (1 << 62)) & (mask64 < (7 << 60))).all())
    assert_equal(wide.reveal(mask[1]), limbs_from_uint64(mask64))
    fresh = extension_material(base, wide, values.shape)
    assert_equal(wide.reveal(extend_online(base, wide, share, fresh)), limbs_from_uint64(values))
    # Include signed extrema of the supported truncation interval and residues.
    x[:8] = torch.tensor([-(1 << 62), -(1 << 62)+1, -3, -1, 0, 1, 3, (1 << 62)-1])
    for shift in [1, 18, 24, 62]:
        material = prob_trunc_material(base, x.shape, shift)
        sx = base.private(x, 0)
        result = base.reveal(prob_trunc(base, sx, shift, material))
        lower = x >> shift
        if not bool(((result == lower) | (result == lower + 1)).all()):
            raise AssertionError("probabilistic truncation outside adjacent rounding values")
    # Independent random scales and one-shot enforcement, outside benchmark.
    for mu in [.25, 4.]:
        points = torch.tensor([[-8., -1., 0., 1., 8.]], dtype=torch.float64)
        points_q17 = (points * (1 << 17)).long()
        output = ruler_g2(base, wide, base.private(points_q17, 0), prepare_g2(base, wide, points.shape, mu))
        actual = base.reveal(output).double() / (1 << 17)
        if error_metrics(actual, g2(points, mu))["max_abs"] >= .005:
            raise AssertionError("fixed-point boundary test failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=257)
    parser.add_argument("--primitive-count", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mu", type=float, default=MU0)
    parser.add_argument("--crypten", action="store_true")
    args = parser.parse_args()
    if args.count < 3 or args.primitive_count < 8 or args.repeats < 1 or not .25 <= args.mu <= 4:
        parser.error("require count>=3, primitive-count>=8, repeats>=1, and .25<=mu<=4")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Temporary store stays under the agreed output root, not system /tmp.
    with tempfile.TemporaryDirectory(prefix="gloo-", dir=args.output) as directory:
        rendezvous = Path(directory, "store").as_uri()
        options = vars(args).copy()
        options["output"] = str(args.output)
        mp.spawn(worker, args=(rendezvous, options), nprocs=3, join=True)


if __name__ == "__main__":
    main()
