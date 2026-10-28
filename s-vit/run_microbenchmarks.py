"""Run complete three-party CUDA microbenchmarks and build paper tables.

Each shape runs in an isolated process group. Raw JSON reports are preserved
under the caller-supplied output root, and a backend-specific table is emitted
only after every required shape and method has completed.
"""

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import microbenchmark_table as table


def _run_case(command, timeout_seconds, backend):
    environment = os.environ.copy()
    if backend == "nccl":
        environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    print("\nRunning: " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"Case exceeded {timeout_seconds}s; terminating its process group.",
            file=sys.stderr,
            flush=True,
        )
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(f"benchmark case timed out after {timeout_seconds}s")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_operator(operator, backend, root, repeats, distributions, timeout_seconds):
    config = table.CONFIG[operator]
    output = root / operator
    cases = [(shape, output / "x".join(str(x) for x in shape))
             for shape in config["shapes"]]
    existing = [directory / "comparison.json" for _, directory in cases
                if (directory / "comparison.json").exists()]
    if existing:
        raise RuntimeError(
            f"existing reports would be overwritten: {existing}; choose a new output root"
        )

    reports = []
    for shape, directory in cases:
        shape_text = "x".join(str(x) for x in shape)
        module = f"{operator}.benchmark_crypten_cuda"
        shape_arg = "--shape" if operator == "sgelu" else "--shapes"
        command = [sys.executable, "-m", module, "--backend", backend,
                   shape_arg, shape_text, "--repeats", str(repeats), "--methods"]
        command.extend(method for method, _, _ in config["methods"])
        if operator == "slayernorm":
            command.extend(["--distributions", *distributions])
        command.extend(["--output", str(directory)])
        print(f"\n[{backend.upper()}] {operator} shape {shape_text}", flush=True)
        _run_case(command, timeout_seconds, backend)
        report = directory / "comparison.json"
        if not report.is_file():
            raise RuntimeError(f"benchmark finished without {report}")
        reports.append(report)

    table_distributions = distributions if operator == "slayernorm" else ["normal"]
    for distribution in table_distributions:
        values, metadata = table.load_measurements(
            reports, operator, distribution, "system_bytes"
        )
        target = output / f"{operator}_microbenchmark_{distribution}.md"
        if target.exists():
            raise RuntimeError(f"table already exists: {target}")
        target.write_text(
            table.render(operator, values, metadata, distribution, "system_bytes"),
            encoding="utf-8",
        )
        print(f"Table: {target}", flush=True)


def main(default_backend="nccl"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default=default_backend)
    parser.add_argument("--operator", choices=("sgelu", "slayernorm", "both"),
                        required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-timeout", type=int, default=1800,
                        help="maximum seconds per shape (default: 1800)")
    parser.add_argument("--distributions", nargs="+", choices=("normal", "uniform"),
                        default=["normal", "uniform"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.case_timeout < 1:
        parser.error("case-timeout must be positive")
    if len(set(args.distributions)) != len(args.distributions):
        parser.error("distributions must not contain duplicates")
    args.output_root.mkdir(parents=True, exist_ok=True)
    operators = ("sgelu", "slayernorm") if args.operator == "both" else (args.operator,)
    try:
        for operator in operators:
            run_operator(operator, args.backend, args.output_root, args.repeats,
                         args.distributions, args.case_timeout)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\nCompleted reports were preserved.\n")


if __name__ == "__main__":
    main()
