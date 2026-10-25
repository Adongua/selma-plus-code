"""Synthetic three-party operator benchmark; every phase uses the same counters.

Run: python -m slayernorm.benchmark --output ../tmp/slayernorm
Shapes denote (batch, tokens, hidden_width); normalization is over the last
hidden-width axis for every (batch, token) row.
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import platform
import sys
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from sgelu.runtime import Ring3, Ring3Wide
from .protocol import METHODS, Profile, prepare, layernorm

ROOT = Path(__file__).resolve().parents[3]


def start_parties(worker, rendezvous, options):
    """Start CPU/Gloo parties without reinitializing macOS OpenMP in children."""
    method = "fork" if sys.platform == "darwin" else "spawn"
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "en0")
    mp.start_processes(worker, args=(rendezvous, options), nprocs=3,
                       join=True, start_method=method)


def fixture(shape, distribution, seed):
    rng = torch.Generator().manual_seed(seed)
    if distribution == "normal":
        x = torch.randn(shape, generator=rng, dtype=torch.float64).clamp(-4, 4)
    elif distribution == "uniform":
        x = torch.rand(shape, generator=rng, dtype=torch.float64) * 8 - 4
    elif distribution == "constant":
        x = torch.full(shape, 1.25, dtype=torch.float64)
    else:
        x = 1.25 + .001 * torch.randn(shape, generator=rng, dtype=torch.float64)
    gamma = torch.linspace(.75, 1.25, shape[-1], dtype=torch.float64)
    beta = torch.linspace(-.1, .1, shape[-1], dtype=torch.float64)
    encode = lambda v: (v * 2**17).round().long()
    return tuple(encode(v) for v in (x, gamma, beta))


def errors(value, target):
    delta = (value - target).abs().flatten()
    if not torch.isfinite(delta).all():
        raise ArithmeticError("nonfinite output")
    return {"mae": delta.mean().item(), "max_abs": delta.max().item(),
            "rmse": delta.square().mean().sqrt().item(),
            "p99_abs": torch.quantile(delta, .99).item()}


def measure(base, fn):
    dist.barrier()
    sent, phases = base.sent_bytes, base.phases
    started = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - started
    metrics = torch.tensor([seconds, base.sent_bytes-sent, base.phases-phases], dtype=torch.float64)
    gathered = [torch.empty_like(metrics) for _ in range(3)]
    dist.all_gather(gathered, metrics)
    metrics = torch.stack(gathered)
    return result, {"seconds_max": metrics[:, 0].max().item(),
                    "sent_bytes_sum": int(metrics[:, 1].sum()),
                    "exchange_phases_per_rank": metrics[:, 2].int().tolist()}


def worker(rank, rendezvous, options):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=300))
    try:
        base, records = Ring3(), []
        wide = Ring3Wide(base)
        profile = Profile(epsilon=options["epsilon"], scale=options["scale"])
        for shape in options["shapes"]:
            profile.validate(shape)
            for distribution in options["distributions"]:
                x, gamma, beta = fixture(shape, distribution, options["seed"])
                xf, gf, bf = [v.double() / 2**17 for v in (x, gamma, beta)]
                target = F.layer_norm(xf, (shape[-1],), gf, bf, profile.epsilon)
                effective_epsilon = profile.epsilon_integer / 2**16 * profile.scale**2
                effective_target = F.layer_norm(xf, (shape[-1],), gf, bf, effective_epsilon)
                for iteration in range(options["repeats"] + 1):
                    # Rotate execution order to reduce fixed-order timing bias.
                    methods = options["methods"]
                    for offset in range(len(methods)):
                        method = methods[(iteration + offset) % len(methods)]
                        shares, input_metrics = measure(base, lambda: (
                            base.private(x, 0), base.private(gamma, 1), base.private(beta, 2)))
                        material, offline = measure(base, lambda: prepare(base, wide, shape, profile, method))
                        output, online = measure(base, lambda: layernorm(base, wide, *shares, material))
                        opened, reconstruction = measure(base, lambda: base.reveal(output))
                        # Ensure accidental reuse cannot execute any further communication.
                        try:
                            layernorm(base, wide, *shares, material)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError("one-use material guard failed")
                        if rank == 0:
                            result = opened.double() / 2**17
                            error = errors(result, target)
                            if distribution in ("normal", "uniform") and error["max_abs"] > .005:
                                raise AssertionError(f"{method} ordinary-input error exceeds 0.005: {error}")
                            records.append({"shape": list(shape), "distribution": distribution,
                                "method": method, "warmup": iteration == 0,
                                "input_sharing": input_metrics, "offline": offline,
                                "online": online, "output_reconstruction": reconstruction,
                                "error_vs_torch": error,
                                "meets_max_abs_0_005": error["max_abs"] <= .005,
                                "error_vs_effective_epsilon": errors(result, effective_target),
                                "variance_truncation_elements": math.prod(shape[:-1]) *
                                    (shape[-1] + 1 if method in ("unfused_ruler", "unfused_newton") else 1),
                                "input_min": xf.min().item(), "input_max": xf.max().item()})
        if rank == 0:
            report = {"status": "executed_not_security_validated", "profile": vars(profile),
                "q_input": 17, "q_variance": 16, "q_inverse": 24,
                "effective_epsilon": effective_epsilon, "seed": options["seed"],
                "repeats": options["repeats"], "world_size": 3, "backend": "Gloo / Ring3",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "platform": platform.platform(), "threads_per_party": 1},
                "communication": "sum of actual Ring3 sent payload bytes across 3 parties; receive bytes not double-counted; excludes barriers, metric gathers and runtime startup",
                "security": "Ruler opens a^2 V; bounded multiplicative masking and share extension are not proven private. Newton is a same-runtime implementation baseline, not a literature SOTA claim.",
                "records": records}
            output_path = Path(options["output"])
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            (output_path / "summary.md").write_text(summarize(report))
            print(json.dumps({"report": str(output_path / "comparison.json"),
                              "cases": len(records), "summary": str(output_path / "summary.md")}))
    finally:
        dist.destroy_process_group()


def summarize(report):
    lines = ["# sLayerNorm 算子 micro benchmark", "",
        "三方本机 CPU/Gloo；shape=(batch, token 数, 隐藏维度)，每个 (batch, token) 行按隐藏维归一化。输入和仿射参数均为秘密分享。",
        f"每组预热一次，正式重复 {report['repeats']} 次。时间取每次三方最慢值，再求平均；通信为三方发送载荷合计，MiB=2^20 bytes。",
        "在线时间不包含输入分享、离线材料生成、输出重构与同步；这些阶段均在 JSON 中分别记录。", "",
        "| 分布 | Shape | 方法 | 在线(s) | 离线(s) | 在线(MiB) | 离线(MiB) | MAE | 最大绝对误差 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    groups = {}
    for r in report["records"]:
        if not r["warmup"]:
            groups.setdefault((r["distribution"], tuple(r["shape"]), r["method"]), []).append(r)
    for (distribution, shape, method), rows in sorted(groups.items()):
        avg = lambda phase, field: statistics.mean(r[phase][field] for r in rows)
        maximum = max(r["error_vs_torch"]["max_abs"] for r in rows)
        lines.append(f"| {distribution} | {shape} | {method} | {avg('online','seconds_max'):.6f} | {avg('offline','seconds_max'):.6f} | {avg('online','sent_bytes_sum')/2**20:.6f} | {avg('offline','sent_bytes_sum')/2**20:.6f} | {avg('error_vs_torch','mae'):.6g} | {maximum:.6g} |")
    lines += ["", "- fused_ruler：初稿融合方差 + Ruler 倒平方根（实验性）。",
        "- unfused_ruler：逐通道平方截断 + 同一 Ruler，用于隔离方差融合收益。",
        "- fused_newton：相同融合方差 + 32 次秘密 Newton 迭代，初值 0.125；工程对照，未宣称最优迭代次数或 SOTA。",
        f"- 误差参照 torch.float64 LayerNorm，epsilon={report['profile']['epsilon']}；Q16 的实际稳定项折算为原输入尺度是 {report['effective_epsilon']:.10g}。JSON 同时报告相对此稳定项的误差。",
        "- Ruler 公开 a²V 的隐私性未获证明，不能将本表作为相同安全保证下的性能优势结论。",
        "- 随机输入为合成数据，normal=N(0,1) 后裁剪至 [-4,4]；uniform=U[-4,4]；constant=1.25；near_constant=1.25+0.001*N(0,1)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "slayernorm")
    parser.add_argument("--shapes", nargs="+", default=["1x197x768", "1x257x768", "1x577x768", "8x257x768"])
    parser.add_argument("--distributions", nargs="+", choices=["normal", "uniform", "constant", "near_constant"], default=["normal", "uniform"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeat count required")
    try:
        args.shapes = [tuple(int(n) for n in shape.split("x")) for shape in args.shapes]
        for shape in args.shapes:
            Profile(epsilon=args.epsilon, scale=args.scale).validate(shape)
    except ValueError as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
    with tempfile.TemporaryDirectory(prefix="slayernorm-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        start_parties(worker, Path(directory, "store").as_uri(), options)


if __name__ == "__main__":
    main()
