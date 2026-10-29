"""Measure grouped Ring3 head costs and assemble an explicit cost table."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from run_microbenchmarks import _run_case
from .group_costs import assemble_reports
from .run_microbenchmark import parse_shape, shape_text


def parse_pair(value):
    try:
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError
        result = tuple(int(x) for x in parts)
        if any(x < 0 for x in result):
            raise ValueError
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use nonnegative RSA,Mixer counts such as 2,1") from error


def all_pairs(heads):
    return [(rsa, mixer) for rsa in range(heads + 1) for mixer in range(heads + 1 - rsa)]


def validate_report(path, shape, paths, args):
    report = json.loads(path.read_text())
    expected = {"backend": args.backend, "world_size": 3, "shape_b_h_n_d": list(shape),
                "repeats": args.repeats, "seed": args.seed, "input_precision_bits": 17,
                "ring3_precision_bits": 17, "has_cls_token": True,
                "patch_grid": [math.isqrt(shape[2] - 1)] * 2}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key} does not match requested configuration")
    if report.get("head_types", {}).get("hybrid") != paths:
        raise ValueError(f"{path}: head order does not match requested configuration")
    if report.get("methods") != ["hybrid"]:
        raise ValueError(f"{path}: expected a standalone hybrid case")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, required=True, help="B x heads x tokens x head_dim")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pairs", nargs="+", type=parse_pair, help="explicit RSA,Mixer counts")
    group.add_argument("--all-counts", action="store_true", help="measure all (H+1)(H+2)/2 combinations")
    parser.add_argument("--resume", action="store_true", help="reuse matching completed case reports")
    args = parser.parse_args()
    if args.repeats < 1 or args.case_timeout < 1:
        parser.error("repeats and case-timeout must be positive")
    if args.table_output.exists():
        parser.error("table output already exists; choose a new file")
    heads = args.shape[1]
    pairs = all_pairs(heads) if args.all_counts else args.pairs
    if len(set(pairs)) != len(pairs) or any(i + j > heads for i, j in pairs):
        parser.error("pairs must be unique and RSA+Mixer cannot exceed head count")
    root = args.output_root.resolve()
    paths = []
    try:
        for number, (rsa, mixer) in enumerate(pairs, 1):
            directory = root / f"rsa{rsa}_mixer{mixer}"
            report = directory / "comparison.json"
            path_types = ["rsa"] * rsa + ["lsa"] * (heads - rsa - mixer) + ["mixer"] * mixer
            print(json.dumps({"event": "case_started", "case": number, "cases": len(pairs),
                              "rsa": rsa, "mixer": mixer}), flush=True)
            if report.exists() and not args.resume:
                raise ValueError(f"{report} exists; use --resume or a new output root")
            if not report.exists():
                command = [sys.executable, "-m", "hattention.benchmark_compare_cuda",
                           "--backend", args.backend, "--shape", shape_text(args.shape),
                           "--methods", "hybrid", "--head-types", *path_types,
                           "--patch-grid", f"{math.isqrt(args.shape[2] - 1)}x{math.isqrt(args.shape[2] - 1)}",
                           "--repeats", str(args.repeats), "--seed", str(args.seed),
                           "--output", str(directory)]
                _run_case(command, args.case_timeout, args.backend)
            validate_report(report, args.shape, path_types, args)
            paths.append(report)
        table = assemble_reports(paths)
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        with args.table_output.open("x", encoding="utf-8") as target:
            json.dump(table, target, indent=2)
            target.write("\n")
        print(json.dumps({"table": str(args.table_output), "complete": table["complete"],
                          "measured": len(table["entries"]),
                          "missing": len(table["missing_counts"])}), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Calibration stopped: {error}\nCompleted case reports were preserved.\n")


if __name__ == "__main__":
    main()
