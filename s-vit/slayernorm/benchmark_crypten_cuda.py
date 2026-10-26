"""Unified three-party CUDA microbenchmark for sLayerNorm and baselines.

The manuscript sLayerNorm and the Newton engineering baseline use the Ring3
adapter on the initialized NCCL process group.  Literature baselines are
re-evaluated with CrypTen MPCTensor so every baseline has the same GPU,
fixture, affine secrecy, and three-party process topology.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from .baselines import (
    ALL_METHODS,
    CRYPTEN_METHODS,
    METHOD_PROVENANCE,
    METHOD_YEAR,
    crypten_layernorm,
)
from .protocol import Profile, layernorm, prepare


RING_METHODS = {
    "slayernorm": "fused_ruler",
    "newton_engineering": "unfused_newton",
}


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 1x197x256") from error
    if len(shape) != 3 or any(dimension < 1 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must be positive BxTxD")
    return shape


def _fixture(shape, distribution, seed, device):
    generator = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        x = torch.randn(shape, generator=generator, dtype=torch.float64).clamp(-4, 4)
    elif distribution == "uniform":
        x = torch.rand(shape, generator=generator, dtype=torch.float64) * 8 - 4
    elif distribution == "constant":
        x = torch.full(shape, 1.25, dtype=torch.float64)
    else:
        x = 1.25 + 0.001 * torch.randn(
            shape, generator=generator, dtype=torch.float64
        )
    gamma = torch.linspace(0.75, 1.25, shape[-1], dtype=torch.float64)
    beta = torch.linspace(-0.1, 0.1, shape[-1], dtype=torch.float64)
    quantize = lambda tensor: (
        (tensor * (1 << 17)).round() / (1 << 17)
    ).to(device)
    return tuple(quantize(tensor) for tensor in (x, gamma, beta))


def _barrier(device):
    dist.barrier(device_ids=[device.index])


def _sync(device):
    torch.cuda.synchronize(device)


def _aggregate(device, seconds, stats, *, bidirectional_counter=False):
    row = torch.tensor(
        [seconds, stats["bytes"], stats["rounds"], stats["time"]],
        dtype=torch.float64,
        device=device,
    )
    rows = [torch.empty_like(row) for _ in range(3)]
    dist.all_gather(rows, row)
    values = torch.stack(rows).cpu()
    logged_bytes = int(values[:, 1].sum().item())
    return {
        "seconds_max": values[:, 0].max().item(),
        "sent_bytes_sum": logged_bytes,
        "system_bytes": logged_bytes // 2 if bidirectional_counter else logged_bytes,
        "rounds_per_rank": values[:, 2].int().tolist(),
        "comm_time_sum": values[:, 3].sum().item(),
    }


def _crypten_measure(device, crypten, function):
    _barrier(device)
    crypten.reset_communication_stats()
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    stats = crypten.communicator.get().get_communication_stats()
    return result, _aggregate(
        device, seconds, stats, bidirectional_counter=True
    )


def _ring_measure(device, base, function):
    _barrier(device)
    _sync(device)
    bytes_before, rounds_before = base.sent_bytes, base.phases
    started = time.perf_counter()
    result = function()
    _sync(device)
    seconds = time.perf_counter() - started
    return result, _aggregate(device, seconds, {
        "bytes": base.sent_bytes - bytes_before,
        "rounds": base.phases - rounds_before,
        "time": 0.0,
    })


def _zero_phase():
    return {
        "seconds_max": 0.0,
        "sent_bytes_sum": 0,
        "system_bytes": 0,
        "rounds_per_rank": [0, 0, 0],
        "comm_time_sum": 0.0,
    }


def _plain(value):
    return (value.tensor() if hasattr(value, "tensor") else value).double()


def _errors(value, target):
    delta = (value - target).abs().flatten()
    if not torch.isfinite(delta).all():
        raise ArithmeticError("non-finite LayerNorm output")
    return {
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "p99_abs": torch.quantile(delta, 0.99).item(),
    }


class CrypTenCudaRingLayerNorm:
    """Run one existing Ring3 LayerNorm method from CrypTen CUDA shares."""

    def __init__(self, device, profile, ring_method):
        from crypten.cuda import CUDALongTensor
        from crypten.mpc import MPCTensor, ptype

        self.CUDALongTensor = CUDALongTensor
        self.MPCTensor = MPCTensor
        self.ptype = ptype
        self.base = CudaRing3(device=device, reveal_mode="p2p")
        self.wide = CudaRing3Wide(self.base)
        self.profile = profile
        self.ring_method = ring_method
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("pending LayerNorm material must be consumed first")
        self.material = prepare(
            self.base, self.wide, tuple(shape), self.profile, self.ring_method
        )

    def __call__(self, value, gamma, beta):
        for tensor in (value, gamma, beta):
            if tensor.ptype != self.ptype.arithmetic:
                raise ValueError("Ring3 LayerNorm requires arithmetic CrypTen shares")
            if tensor.encoder._precision_bits != 17:
                raise ValueError("Ring3 LayerNorm requires Q17 CrypTen shares")
        if self.material is None:
            raise RuntimeError("prepare LayerNorm material first")
        material, self.material = self.material, None
        output = layernorm(
            self.base,
            self.wide,
            value.share.data,
            gamma.share.data,
            beta.share.data,
            material,
        )
        return self.MPCTensor.from_shares(
            self.CUDALongTensor(output), precision=17
        )


def _summarize(report):
    lines = [
        "# sLayerNorm CUDA micro benchmark",
        "",
        "所有 shape 为 `(B,T,D)`，沿最后一维执行 LayerNorm。时间为三方最慢值；"
        "通信量使用三方系统级发送载荷，CrypTen 的双向日志计数已除以二。",
        "",
        "| 分布 | Shape | 方法 | 在线(s) | 离线(s) | 在线通信(KB) | 在线轮数 | MAE | Max error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = {}
    for record in report["records"]:
        if record["warmup"]:
            continue
        for method, result in record["methods"].items():
            key = (record["distribution"], tuple(record["shape"]), method)
            groups.setdefault(key, []).append(result)
    for (distribution, shape, method), rows in sorted(groups.items()):
        average = lambda phase, field: statistics.mean(
            row[phase][field] for row in rows
        )
        rounds = statistics.mean(
            max(row["online"]["rounds_per_rank"]) for row in rows
        )
        lines.append(
            f"| {distribution} | {shape} | {method} | "
            f"{average('online', 'seconds_max'):.6f} | "
            f"{average('offline', 'seconds_max'):.6f} | "
            f"{average('online', 'system_bytes') / 1024:.3f} | "
            f"{rounds:.1f} | "
            f"{statistics.mean(row['error_vs_torch']['mae'] for row in rows):.6g} | "
            f"{max(row['error_vs_torch']['max_abs'] for row in rows):.6g} |"
        )
    lines += [
        "",
        "CrypTen baseline 的相关材料由其 provider 在算子调用期间取得，因此计入在线阶段；"
        "Ring3 方法显式区分材料准备与在线执行。两类 round 计数代表各自运行时的通信阶段，"
        "可用于同一运行时内比较，但不是形式化的跨框架协议轮数。",
        "",
    ]
    return "\n".join(lines)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="3",
        RENDEZVOUS=rendezvous,
        DISTRIBUTED_BACKEND="nccl",
    )
    import crypten

    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        profile = Profile(epsilon=options["epsilon"], scale=options["scale"])
        ring_operators = {
            method: CrypTenCudaRingLayerNorm(device, profile, ring_method)
            for method, ring_method in RING_METHODS.items()
            if method in options["methods"]
        }
        records = []
        if rank == 0 and options["progress"]:
            print(json.dumps({
                "event": "initialized",
                "framework": "CrypTen MPCTensor + Ring3 / NCCL / CUDA",
                "shapes": [list(shape) for shape in options["shapes"]],
                "methods": options["methods"],
            }), flush=True)

        for shape in options["shapes"]:
            profile.validate(shape)
            for distribution in options["distributions"]:
                x, gamma, beta = _fixture(
                    shape, distribution, options["seed"], device
                )
                target = F.layer_norm(
                    x, (shape[-1],), gamma, beta, options["epsilon"]
                )
                for iteration in range(options["repeats"] + 1):
                    encrypted, input_sharing = _crypten_measure(
                        device,
                        crypten,
                        lambda: (
                            crypten.cryptensor(
                                x, src=0, precision=17, device=device
                            ),
                            crypten.cryptensor(
                                gamma, src=1, precision=17, device=device
                            ),
                            crypten.cryptensor(
                                beta, src=2, precision=17, device=device
                            ),
                        ),
                    )
                    row = {
                        "shape": list(shape),
                        "distribution": distribution,
                        "warmup": iteration == 0,
                        "input_sharing": input_sharing,
                        "methods": {},
                    }
                    methods = options["methods"]
                    ordered = methods[iteration % len(methods):] + methods[:iteration % len(methods)]
                    for method in ordered:
                        x_secret, gamma_secret, beta_secret = encrypted
                        if method in ring_operators:
                            operator = ring_operators[method]
                            _, offline = _ring_measure(
                                device,
                                operator.base,
                                lambda operator=operator: operator.prepare(shape),
                            )
                            output, online = _ring_measure(
                                device,
                                operator.base,
                                lambda operator=operator: operator(
                                    x_secret.clone(),
                                    gamma_secret.clone(),
                                    beta_secret.clone(),
                                ),
                            )
                        else:
                            offline = _zero_phase()
                            output, online = _crypten_measure(
                                device,
                                crypten,
                                lambda method=method: crypten_layernorm(
                                    method,
                                    x_secret.clone(),
                                    gamma_secret.clone(),
                                    beta_secret.clone(),
                                    options["epsilon"],
                                ),
                            )
                        opened, reconstruction = _crypten_measure(
                            device,
                            crypten,
                            lambda output=output: output.get_plain_text(),
                        )
                        if rank == 0:
                            result = _plain(opened)
                            row["methods"][method] = {
                                "offline": offline,
                                "online": online,
                                "output_reconstruction": reconstruction,
                                "error_vs_torch": _errors(result, target),
                                "online_includes_provider": method in CRYPTEN_METHODS,
                            }
                            if options["progress"]:
                                print(json.dumps({
                                    "event": "method_complete",
                                    "shape": list(shape),
                                    "distribution": distribution,
                                    "warmup": iteration == 0,
                                    "method": method,
                                    "online_seconds": online["seconds_max"],
                                    "online_system_bytes": online["system_bytes"],
                                    "rounds_max": max(online["rounds_per_rank"]),
                                }), flush=True)
                    if rank == 0:
                        records.append(row)

        if rank == 0:
            output = Path(options["output"])
            output.mkdir(parents=True, exist_ok=True)
            report = {
                "status": "executed_not_security_validated",
                "framework": "CrypTen-main MPCTensor + Ring3 adapters / NCCL / CUDA",
                "world_size": 3,
                "devices": [f"cuda:{index}" for index in range(3)],
                "shapes": [list(shape) for shape in options["shapes"]],
                "distributions": options["distributions"],
                "methods": options["methods"],
                "repeats": options["repeats"],
                "epsilon": options["epsilon"],
                "scale": options["scale"],
                "seed": options["seed"],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fixture": "synthetic float64 generated on CPU, quantized to Q17 before sharing",
                "method_provenance": METHOD_PROVENANCE,
                "method_year": METHOD_YEAR,
                "communication": (
                    "system_bytes is sent-payload equivalent across three parties; "
                    "CrypTen logs send+receive and is divided by two, while Ring3 "
                    "already counts sends only"
                ),
                "timing_boundary": (
                    "Ring3 methods separate preprocessing; CrypTen provider work "
                    "inside MPCTensor primitives is included in online"
                ),
                "security": (
                    "literature methods are same-framework 3PC adaptations, not "
                    "official runtime numbers; Ruler privacy caveats and CUDA "
                    "benchmark-PRG caveats remain"
                ),
                "records": records,
            }
            report_path = output / "comparison.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            summary_path = output / "summary.md"
            summary_path.write_text(_summarize(report) + "\n")
            print(json.dumps({
                "report": str(report_path),
                "summary": str(summary_path),
                "records": len(records),
            }, indent=2), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes", nargs="+", type=_parse_shape, default=[(1, 197, 256)]
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["normal", "uniform", "constant", "near_constant"],
        default=["normal"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS)
    )
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="disable rank-0 per-method progress messages",
    )
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    if torch.cuda.device_count() < 3:
        parser.error("at least three CUDA devices are required")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("epsilon must be finite and positive")
    try:
        for shape in args.shapes:
            Profile(epsilon=args.epsilon, scale=args.scale).validate(shape)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="crypten-layernorm-cuda-", dir=args.output.parent
    ) as directory:
        options = vars(args).copy()
        options["shapes"] = [tuple(shape) for shape in args.shapes]
        options["methods"] = list(args.methods)
        options["distributions"] = list(args.distributions)
        options["output"] = str(args.output)
        mp.spawn(
            worker,
            args=(Path(directory, "store").as_uri(), options),
            nprocs=3,
            join=True,
        )


if __name__ == "__main__":
    main()
