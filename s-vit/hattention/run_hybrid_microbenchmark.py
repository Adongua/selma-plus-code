"""Compare fixed four-head attention configurations on synthetic inputs."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from run_microbenchmarks import _run_case
from .run_microbenchmark import parse_shape, shape_text


CONFIGS = {
    "standard": ("standard", None),
    "rsa4": ("hybrid", ("rsa", "rsa", "rsa", "rsa")),
    "rsa2_lsa1_mixer1": ("hybrid", ("rsa", "rsa", "lsa", "mixer")),
    "rsa1_lsa1_mixer2": ("hybrid", ("rsa", "lsa", "mixer", "mixer")),
    # Interleaving exercises restoration of the original head order.
    "mixer1_rsa1_lsa1_mixer1": ("hybrid", ("mixer", "rsa", "lsa", "mixer")),
}


def collect(path, shape, name, args):
    report = json.loads(path.read_text(encoding="utf-8"))
    method, paths = CONFIGS[name]
    expected = {"backend": args.backend, "world_size": 3, "shape_b_h_n_d": list(shape),
                "methods": [method], "repeats": args.repeats, "seed": args.seed,
                "input_precision_bits": 17, "ring3_precision_bits": 17,
                "crypten_precision_bits": args.crypten_precision_bits,
                "max_abs_error_threshold": args.max_abs_error, "input_std": 0.25,
                "has_cls_token": True}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key}={report.get(key)!r}, expected {value!r}")
    expected_paths = list(paths) if paths else ["softmax"] * shape[1]
    if report.get("head_types", {}).get(method) != expected_paths:
        raise ValueError(f"{path}: head configuration does not match {name}")
    rows = [row["methods"][method] for row in report["records"] if not row["warmup"]]
    if len(rows) != args.repeats:
        raise ValueError(f"{path}: expected {args.repeats} measured repetitions")
    for row in rows:
        heads = row.get("per_head_error_vs_own_reference", [])
        if len(heads) != shape[1] or any(
            item.get("head") != index or item.get("path") != expected_paths[index]
            for index, item in enumerate(heads)
        ):
            raise ValueError(f"{path}: missing or inconsistent per-head diagnostics")
    if not all(row["correctness_passed"] for row in rows):
        return None
    values = [(row["online"]["seconds_max"], row["online"]["system_bytes"] / 1024,
               max(row["online"]["rounds_per_rank"]),
               row["error_vs_own_reference"]["max_abs"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for row in values for value in row):
        raise ValueError(f"{path}: invalid metrics")
    return (statistics.mean(row[0] for row in values),
            statistics.mean(row[1] for row in values),
            statistics.mean(row[2] for row in values), max(row[3] for row in values))


def render(shapes, values, args):
    lines = ["# Mixed-head attention core benchmark", "",
             f"Backend: {args.backend}; 3 parties; input Q17; CrypTen standard Q{args.crypten_precision_bits}; "
             f"Ring3 Q17; seed {args.seed}; {args.repeats} repetitions after one warmup.", "",
             "| Shape (B,H,N,D) | Configuration (head order) | Online (s) | Comm. (KiB) | Rounds / phases | Max error vs own reference |",
             "|---|---|---:|---:|---:|---:|"]
    for shape in shapes:
        for name, (method, paths) in CONFIGS.items():
            label = "standard softmax x4" if method == "standard" else ", ".join(paths)
            result = values[shape, name]
            cells = ("INVALID",) * 4 if result is None else (
                f"{result[0]:.5f}", f"{result[1]:.1f}", f"{result[2]:.1f}", f"{result[3]:.6g}")
            lines.append(f"| {shape_text(shape)} | {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "Each row uses the same seed and Q17 synthetic Q/K/V; head-path choices are public. "
                  "Errors are against each configuration's own plaintext formula, not standard Attention. "
                  "Every measured repetition must pass the maximum-error threshold; individual head "
                  "errors and offline costs are available in each raw JSON report.", "",
                  "Standard includes CrypTen provider work online; Ring3 preprocessing is excluded. "
                  "CrypTen counts communicator rounds and Ring3 counts exchange phases: these are "
                  "different implementation counters. Communication is an estimate. Gloo stages "
                  "Ring3 shares through CPU. These are attention-core results, excluding projections "
                  "and full ViT; Ruler privacy is not validated."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=[parse_shape("1x4x17x16")])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--crypten-precision-bits", type=int, default=20)
    parser.add_argument("--max-abs-error", type=float, default=0.005)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(shape[1] != 4 for shape in args.shapes):
        parser.error("all shapes must have exactly four heads")
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("duplicate shapes are not allowed")
    if (args.repeats < 1 or args.case_timeout < 1 or not 8 <= args.crypten_precision_bits <= 24
            or not math.isfinite(args.max_abs_error) or args.max_abs_error <= 0):
        parser.error("invalid repeats, timeout, precision or error threshold")
    root = args.output_root.resolve()
    target = root / "hattention_hybrid_microbenchmark.md"
    cases = [(shape, name, root / shape_text(shape) / name)
             for shape in args.shapes for name in CONFIGS]
    if target.exists():
        parser.error(f"table already exists: {target}; choose a new output root")
    if not args.resume and any((directory / "comparison.json").exists() for _, _, directory in cases):
        parser.error("existing case reports found; use a new output root or --resume")
    values, environment = {}, None
    try:
        for shape, name, directory in cases:
            method, paths = CONFIGS[name]
            path = directory / "comparison.json"
            if not path.exists():
                command = [sys.executable, "-m", "hattention.benchmark_compare_cuda",
                           "--backend", args.backend, "--shape", shape_text(shape),
                           "--methods", method, "--repeats", str(args.repeats),
                           "--seed", str(args.seed), "--crypten-precision-bits",
                           str(args.crypten_precision_bits), "--max-abs-error",
                           str(args.max_abs_error), "--allow-invalid-report", "--output", str(directory)]
                if paths:
                    command.extend(["--head-types", *paths])
                _run_case(command, args.case_timeout, args.backend)
            values[shape, name] = collect(path, shape, name, args)
            report = json.loads(path.read_text(encoding="utf-8"))
            current = tuple(report.get(key) for key in ("crypten_import", "python", "torch"))
            if not all(current) or (environment is not None and current != environment):
                raise ValueError(f"{path}: missing or inconsistent environment metadata")
            environment = current
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(args.shapes, values, args), encoding="utf-8")
        print(f"Table: {target}", flush=True)
        if any(value is None for value in values.values()):
            print("Some cases failed correctness and are marked INVALID; inspect raw reports.", file=sys.stderr)
            raise SystemExit(1)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\nCompleted reports were preserved; use --resume.\n")


if __name__ == "__main__":
    main()
