"""Run full three-party Gloo/CUDA paper-table experiments, then build tables.

Requires three CUDA GPUs and the project's CrypTen dependencies. Outputs are
written only beneath the explicitly supplied output root.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import microbenchmark_table as table


def run_operator(operator, root, repeats, distributions):
    config = table.CONFIG[operator]
    output = root / operator
    cases = [(shape, output / "x".join(str(x) for x in shape))
             for shape in config["shapes"]]
    existing = [directory / "comparison.json" for _, directory in cases
                if (directory / "comparison.json").exists()]
    if existing:
        raise RuntimeError(f"existing reports would be overwritten: {existing}; choose a new output root")

    reports = []
    for shape, directory in cases:
        shape_text = "x".join(str(x) for x in shape)
        module = f"{operator}.benchmark_crypten_cuda"
        shape_arg = "--shape" if operator == "sgelu" else "--shapes"
        command = [sys.executable, "-m", module, "--backend", "gloo", shape_arg,
                   shape_text, "--repeats", str(repeats), "--methods"]
        command.extend(method for method, _, _ in config["methods"])
        if operator == "slayernorm":
            command.extend(["--distributions", *distributions])
        command.extend(["--output", str(directory)])
        print("\nRunning: " + shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        report = directory / "comparison.json"
        if not report.is_file():
            raise RuntimeError(f"benchmark finished without {report}")
        reports.append(report)

    table_distributions = distributions if operator == "slayernorm" else ["normal"]
    for distribution in table_distributions:
        values, metadata = table.load_measurements(reports, operator, distribution, "system_bytes")
        target = output / f"{operator}_microbenchmark_{distribution}.md"
        if target.exists():
            raise RuntimeError(f"table already exists: {target}")
        target.write_text(table.render(operator, values, metadata, distribution, "system_bytes"),
                          encoding="utf-8")
        print(f"Table: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("sgelu", "slayernorm", "both"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--distributions", nargs="+", choices=("normal", "uniform"),
                        default=["normal", "uniform"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if len(set(args.distributions)) != len(args.distributions):
        parser.error("distributions must not contain duplicates")
    args.output_root.mkdir(parents=True, exist_ok=True)
    operators = ("sgelu", "slayernorm") if args.operator == "both" else (args.operator,)
    try:
        for operator in operators:
            run_operator(operator, args.output_root, args.repeats, args.distributions)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\nCompleted reports were preserved.\n")


if __name__ == "__main__":
    main()
