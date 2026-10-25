"""CrypTen baseline benchmark for the original secure LayerNorm formula.

This is intentionally separate from benchmark.py: sLayerNorm uses the local
Ring3 prototype, while this baseline measures CrypTen's standard secure tensor
primitives under the same three-party Gloo setting.  The report therefore
answers the system-level comparison and records the implementation boundary.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import statistics
import tempfile
import time

import crypten
import crypten.communicator as comm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[3]


def start_parties(worker, rendezvous, options):
    """Use fork on macOS to avoid the unavailable /dev/shm OpenMP path."""
    method = "fork" if sys.platform == "darwin" else "spawn"
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "en0")
    mp.start_processes(worker, args=(rendezvous, options), nprocs=3,
                       join=True, start_method=method)


def fixture(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=generator, dtype=torch.float64).clamp(-4, 4)
    gamma = torch.linspace(.75, 1.25, shape[-1], dtype=torch.float64)
    beta = torch.linspace(-.1, .1, shape[-1], dtype=torch.float64)
    return x, gamma, beta


def secure_layernorm(x, gamma, beta, epsilon):
    """CrypTen composition corresponding to ordinary LayerNorm."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    inverse_std = variance.add(epsilon).inv_sqrt()
    return centered * inverse_std * gamma + beta


def measure(fn):
    communicator = comm.get()
    communicator.barrier()
    before = communicator.get_communication_stats()
    started = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - started
    after = communicator.get_communication_stats()
    local = torch.tensor([seconds, after["bytes"] - before["bytes"],
                          after["rounds"] - before["rounds"]], dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(communicator.get_world_size())]
    dist.all_gather(gathered, local)
    metrics = torch.stack(gathered)
    return value, {
        "seconds_max": metrics[:, 0].max().item(),
        "sent_bytes_sum": int(metrics[:, 1].sum().item()),
        "rounds_max": int(metrics[:, 2].max().item()),
    }


def error(value, target):
    delta = (value - target).abs()
    return {
        "mae": delta.mean().item(),
        "max_abs": delta.max().item(),
        "rmse": delta.square().mean().sqrt().item(),
    }


def run(options):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank = comm.get().get_rank()
    records = []
    epsilon = options["epsilon"]
    for shape in options["shapes"]:
        x, gamma, beta = fixture(shape, options["seed"])
        target = F.layer_norm(x, (shape[-1],), gamma, beta, epsilon)
        for iteration in range(options["repeats"] + 1):
            # CrypTen's normal private-input path; the three tensors originate
            # from different parties to avoid a public-input shortcut.
            shares, sharing = measure(lambda: (
                crypten.cryptensor(x, src=0),
                crypten.cryptensor(gamma, src=1),
                crypten.cryptensor(beta, src=2),
            ))
            output, online = measure(lambda: secure_layernorm(
                shares[0], shares[1], shares[2], epsilon))
            opened, reconstruction = measure(lambda: output.get_plain_text())
            if rank == 0:
                records.append({
                    "shape": list(shape),
                    "warmup": iteration == 0,
                    "input_sharing": sharing,
                    "online": online,
                    "output_reconstruction": reconstruction,
                    "error_vs_torch": error(opened, target),
                })
    if rank == 0:
        report = {
            "status": "executed",
            "method": "CrypTen standard secure LayerNorm composition",
            "formula": "mean -> center -> mean(square) -> inv_sqrt -> affine",
            "epsilon": epsilon,
            "seed": options["seed"],
            "repeats": options["repeats"],
            "world_size": comm.get().get_world_size(),
            "backend": "CrypTen / Gloo",
            "communication": "sum of CrypTen communicator byte counters across parties; excludes metric gather and startup",
            "records": records,
        }
        output = Path(options["output"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "summary.md").write_text(summarize(report) + "\n")
        print(json.dumps({"report": str(output / "comparison.json"), "cases": len(records)}))


def worker(rank, rendezvous, options):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    try:
        crypten.init()
        crypten.cfg.communicator.verbose = True
        run(options)
    finally:
        crypten.uninit()


def summarize(report):
    lines = [
        "# CrypTen 原始 LayerNorm baseline micro benchmark", "",
        "三方本机 CPU/Gloo；shape=(batch, token 数, 隐藏维度)，每个 (batch, token) 行按最后一维归一化。",
        "CrypTen baseline 使用秘密 mean、square、mean、inv_sqrt 和 affine 组合，不包含明文 LayerNorm shortcut。",
        "", "| Shape | 在线(s) | 在线通信(MiB) | rounds | MAE | 最大绝对误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    grouped = {}
    for record in report["records"]:
        if not record["warmup"]:
            grouped.setdefault(tuple(record["shape"]), []).append(record)
    for shape, rows in sorted(grouped.items()):
        avg = lambda field: statistics.mean(r["online"][field] for r in rows)
        max_error = max(r["error_vs_torch"]["max_abs"] for r in rows)
        lines.append(
            f"| {shape} | {avg('seconds_max'):.6f} | "
            f"{avg('sent_bytes_sum') / 2**20:.6f} | "
            f"{avg('rounds_max'):.0f} | "
            f"{statistics.mean(r['error_vs_torch']['mae'] for r in rows):.6g} | "
            f"{max_error:.6g} |"
        )
    lines += ["", "注：与 sLayerNorm 的倍数比较需使用相同 shape、相同重复次数，并明确两者运行时分别是 CrypTen 与 Ring3 原型；该差异属于系统级比较口径。"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "tmp" / "slayernorm-crypten-baseline")
    parser.add_argument("--shapes", nargs="+", default=["1x197x768", "1x257x768", "1x577x768", "8x257x768"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    args.shapes = [tuple(int(value) for value in shape.split("x")) for shape in args.shapes]
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
    with tempfile.TemporaryDirectory(prefix="crypten-ln-", dir=args.output.parent) as directory:
        options = vars(args).copy()
        options["output"] = str(args.output)
        start_parties(worker, Path(directory, "store").as_uri(), options)


if __name__ == "__main__":
    main()
