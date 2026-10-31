"""Remeasure the grouped attention cores used by selected ranked candidates."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

from run_microbenchmarks import _run_case
from .calibrate_group_costs import validate_report
from .group_costs import assemble_reports
from .run_microbenchmark import parse_shape, shape_text


def extract_candidates(report, counts, heads):
    if report.get("status") != "completed_plaintext_not_security_validated":
        raise ValueError("ranked candidate report is not complete")
    candidates = {}
    for row in report["candidates"]:
        count = row["mixer_count"]
        if count in candidates:
            raise ValueError(f"duplicate candidate: {count}")
        layers = row["head_types"]
        if not layers or any(len(layer) != heads or set(layer) - {"rsa", "lsa", "mixer"}
                             for layer in layers):
            raise ValueError(f"invalid layer paths for candidate {count}")
        if sum(layer.count("mixer") for layer in layers) != count:
            raise ValueError(f"mixer count mismatch for candidate {count}")
        candidates[count] = [tuple(layer) for layer in layers]
    missing = set(counts) - candidates.keys()
    if missing:
        raise ValueError(f"missing candidates: {sorted(missing)}")
    selected = {count: candidates[count] for count in counts}
    if len({len(layers) for layers in selected.values()}) != 1:
        raise ValueError("candidate layer counts differ")
    return selected


def group_pair(layer):
    return (layer.count("rsa"), layer.count("mixer"))


def build_summary(candidates, rounds):
    """Sum each independent round's group costs; this is not model latency."""
    results = []
    for count, layers in candidates.items():
        pairs = [group_pair(layer) for layer in layers]
        per_round = []
        for measurements in rounds:
            total = {name: sum(measurements[p][name] for p in pairs)
                     for name in ("online_seconds", "online_bytes", "online_phases")}
            per_round.append(total)
        results.append({"mixer_count": count, "layer_counts": [list(p) for p in pairs],
                        "round_estimates": per_round,
                        "mean_estimate": {k: statistics.mean(row[k] for row in per_round)
                                          for k in per_round[0]},
                        "median_online_seconds": statistics.median(row["online_seconds"] for row in per_round)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mixer-counts", type=int, nargs="+", default=[0, 4, 24])
    parser.add_argument("--shape", type=parse_shape, default=(1, 12, 257, 16))
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print required combinations without running")
    args = parser.parse_args()
    if args.rounds < 1 or args.case_timeout < 1 or len(set(args.mixer_counts)) != len(args.mixer_counts):
        parser.error("rounds and timeout must be positive; mixer counts must be unique")
    grid = math.isqrt(args.shape[2] - 1)
    if grid * grid != args.shape[2] - 1:
        parser.error("shape must have a CLS token and a square patch grid")
    raw = args.candidate_report.read_bytes()
    try:
        candidates = extract_candidates(json.loads(raw), args.mixer_counts, args.shape[1])
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    pairs = sorted({group_pair(layer) for layers in candidates.values() for layer in layers})
    config = {"candidate_report_sha256": hashlib.sha256(raw).hexdigest(),
              "mixer_counts": args.mixer_counts, "shape": list(args.shape), "backend": args.backend,
              "rounds": args.rounds, "seed": args.seed, "pairs": [list(pair) for pair in pairs],
              "case_timeout": args.case_timeout}
    print(json.dumps({"event": "plan", **config, "cases": len(pairs) * args.rounds}), flush=True)
    if args.dry_run:
        return
    root = args.output_root.resolve()
    manifest = root / "run.json"
    if args.resume:
        if not manifest.is_file() or json.loads(manifest.read_text()) != config:
            parser.error("resume requires matching run.json and parameters")
        if (root / "summary.json").exists():
            parser.error("summary already exists; run is complete")
    else:
        root.mkdir(parents=True, exist_ok=False)
        with manifest.open("x") as target:
            json.dump(config, target, indent=2)
            target.write("\n")
    options = SimpleNamespace(backend=args.backend, repeats=1, seed=args.seed)
    round_tables, reference_metadata, sources = [], None, []
    try:
        for round_index in range(args.rounds):
            # Reverse on alternate rounds to reduce systematic order effects.
            order = pairs if round_index % 2 == 0 else list(reversed(pairs))
            paths = []
            for index, (rsa, mixer) in enumerate(order, 1):
                output = root / f"round-{round_index + 1:03d}" / f"rsa{rsa}_mixer{mixer}"
                path = output / "comparison.json"
                head_types = ["rsa"] * rsa + ["lsa"] * (args.shape[1] - rsa - mixer) + ["mixer"] * mixer
                print(json.dumps({"event": "case_started", "round": round_index + 1,
                                  "case": index, "cases": len(pairs), "rsa": rsa, "mixer": mixer}), flush=True)
                if not path.exists():
                    command = [sys.executable, "-m", "hattention.benchmark_compare_cuda", "--backend", args.backend,
                               "--shape", shape_text(args.shape), "--methods", "hybrid", "--head-types", *head_types,
                               "--patch-grid", f"{grid}x{grid}", "--repeats", "1", "--seed", str(args.seed),
                               "--output", str(output)]
                    _run_case(command, args.case_timeout, args.backend)
                validate_report(path, args.shape, head_types, options)
                paths.append(path)
            table = assemble_reports(paths)
            if reference_metadata is not None and table["metadata"] != reference_metadata:
                raise ValueError("measurement metadata differs between rounds")
            reference_metadata = table["metadata"]
            round_tables.append({(e["rsa_heads"], e["mixer_heads"]): e["metrics"] for e in table["entries"]})
            sources.append(table["entries"])
        result = {"status": "completed_estimated_not_security_validated", "config": config,
                  "measurement_metadata": reference_metadata, "group_measurements_by_round": sources,
                  "candidates": build_summary(candidates, round_tables),
                  "notes": "Summed separately measured canonical group timings, not end-to-end or paired statistical "
                           "confidence intervals. Reuse identical group estimates for all matching layers. "
                           "Actual head order, GPU contention, time drift and full-model operations can change cost."}
        with (root / "summary.json").open("x") as target:
            json.dump(result, target, indent=2)
            target.write("\n")
        print(json.dumps({"event": "complete", "summary": str(root / "summary.json"),
                          "candidates": result["candidates"]}), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Measurement stopped: {error}\nCompleted case reports were preserved; use --resume.\n")


if __name__ == "__main__":
    main()
