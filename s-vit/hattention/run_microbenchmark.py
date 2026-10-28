"""Run and summarize four attention-core CUDA microbenchmarks."""

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from run_microbenchmarks import _run_case

METHODS = ("standard", "rsa", "lsa", "mixer")
DEFAULT_SHAPES = ("1x4x197x64", "1x4x257x64", "1x4x577x64", "8x4x257x64")


def parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be BxHxNxD") from error
    if len(shape) != 4 or any(part <= 0 for part in shape):
        raise argparse.ArgumentTypeError("shape must contain four positive dimensions BxHxNxD")
    patches = shape[2] - 1
    if patches < 1 or math.isqrt(patches) ** 2 != patches:
        raise argparse.ArgumentTypeError("Mixer needs N-1 to be a square number (one CLS token)")
    return shape


def shape_text(shape):
    return "x".join(map(str, shape))


def collect_report(path, shape, method, args):
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {"backend": args.backend, "world_size": 3, "shape_b_h_n_d": list(shape),
                "methods": [method], "repeats": args.repeats, "seed": args.seed,
                "input_precision_bits": 17, "crypten_precision_bits": args.crypten_precision_bits,
                "ring3_precision_bits": 17, "max_abs_error_threshold": args.max_abs_error,
                "input_std": 0.25, "has_cls_token": True}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"{path}: {key}={report.get(key)!r}, expected {value!r}")
    rows = [record["methods"][method] for record in report["records"] if not record["warmup"]]
    if len(rows) != args.repeats:
        raise ValueError(f"{path}: expected {args.repeats} measured repetitions")
    if not all(row["correctness_passed"] for row in rows):
        return None
    values = [(row["online"]["seconds_max"], row["online"]["system_bytes"] / 1024,
               max(row["online"]["rounds_per_rank"])) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for row in values for value in row):
        raise ValueError(f"{path}: invalid online metrics")
    return tuple(statistics.mean(row[index] for row in values) for index in range(3))


def render(shapes, values, args):
    header = ["Method"]
    for shape in shapes:
        name = ",".join(map(str, shape))
        header.extend((f"({name}) Time (s)", "Comm. (KiB)", "Phases/rank"))
    lines = ["# Attention core microbenchmark", "",
             f"Backend: {args.backend}; three parties; input Q17; CrypTen standard Q{args.crypten_precision_bits}; "
             f"Ring3 Q17; {args.repeats} measured repetitions, one warmup excluded; seed {args.seed}.", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for method in METHODS:
        cells = [method]
        for shape in shapes:
            result = values[shape, method]
            cells.extend(("INVALID", "INVALID", "INVALID") if result is None else
                         (f"{result[0]:.5f}", f"{result[1]:.1f}", f"{result[2]:.1f}"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Online time is the maximum across parties, then averaged over measured repetitions; "
                  "communication is estimated three-party sent-payload equivalent. "
                  "Phases are the maximum per-party counter, then averaged.", "",
                  "INVALID means at least one measured output failed its own plaintext reference check; "
                  "details, including error metrics and offline measurements, are in the raw JSON.", "",
                  "Standard uses CrypTen native softmax with provider work included online; the other "
                  "paths use experimental Ring3 with separately generated offline material. When using Gloo, "
                  "Ring3 stages through CPU. Framework phase counters and communication estimates differ; "
                  "these are implementation measurements, not identical protocol-round definitions.", "",
                  "Each method uses its own attention formula. Scope: Q/K/V to head output; no projections "
                  "or complete ViT. Ruler privacy has not been validated."])
    return "\n".join(lines) + "\n"


def render_latex(shapes, values, args):
    columns = "l" + "ccc" * len(shapes)
    headings = " & ".join(r"\multicolumn{3}{c}{Shape $(" + ",".join(map(str, shape)) + r")$}" for shape in shapes)
    subheads = " & ".join([r"Time (s) & Comm. (KiB) & Phases"] * len(shapes))
    lines = [r"% Requires booktabs and graphicx", r"\begin{table*}[t]", r"\centering",
             r"\caption{Online cost of attention cores on synthetic inputs.}",
             r"\scriptsize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + columns + "}",
             r"\toprule", "Method & " + headings + r" \\", " & " + subheads + r" \\", r"\midrule"]
    names = {"standard": f"CrypTen standard (Q{args.crypten_precision_bits})",
             "rsa": "RSA", "lsa": "LSA", "mixer": "Mixer"}
    for method in METHODS:
        cells = [names[method]]
        for shape in shapes:
            value = values[shape, method]
            cells.extend(("INVALID", "INVALID", "INVALID") if value is None else
                         (f"{value[0]:.5f}", f"{value[1]:.1f}", f"{value[2]:.1f}"))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\vspace{2pt}",
                  r"\parbox{\textwidth}{\footnotesize Input is Q17; CrypTen standard uses Q"
                  + str(args.crypten_precision_bits) + r" and includes provider work online. "
                  r"Ring3 paths use Q17 with offline preprocessing excluded. Phase counts and "
                  r"communication are framework-specific estimates. INVALID denotes a failed "
                  r"correctness check; see the raw reports for details.}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=[parse_shape(s) for s in DEFAULT_SHAPES])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--crypten-precision-bits", type=int, default=20)
    parser.add_argument("--max-abs-error", type=float, default=0.005)
    parser.add_argument("--case-timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true", help="reuse matching completed case reports")
    args = parser.parse_args()
    if args.repeats < 1 or args.case_timeout < 1 or not 8 <= args.crypten_precision_bits <= 24:
        parser.error("repeats and case-timeout must be positive; precision must be in [8,24]")
    if not math.isfinite(args.max_abs_error) or args.max_abs_error <= 0:
        parser.error("max-abs-error must be finite and positive")
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("shapes must not contain duplicates")
    root = args.output_root.resolve()
    target = root / "hattention_microbenchmark.md"
    cases = [(shape, method, root / shape_text(shape) / method) for shape in args.shapes for method in METHODS]
    latex_target = root / "hattention_microbenchmark.tex"
    if target.exists() or latex_target.exists():
        parser.error(f"table already exists: {target}; choose a new output root")
    if not args.resume and any((directory / "comparison.json").exists() for _, _, directory in cases):
        parser.error("existing case reports found; choose a new output root or use --resume")
    values = {}
    environment = None
    try:
        for shape, method, directory in cases:
            path = directory / "comparison.json"
            if not path.exists():
                command = [sys.executable, "-m", "hattention.benchmark_compare_cuda",
                    "--backend", args.backend, "--shape", shape_text(shape), "--methods", method,
                    "--repeats", str(args.repeats), "--seed", str(args.seed),
                    "--crypten-precision-bits", str(args.crypten_precision_bits),
                    "--max-abs-error", str(args.max_abs_error), "--allow-invalid-report",
                    "--output", str(directory)]
                _run_case(command, args.case_timeout, args.backend)
            values[shape, method] = collect_report(path, shape, method, args)
            report = json.loads(path.read_text(encoding="utf-8"))
            current = tuple(report.get(key) for key in ("crypten_import", "python", "torch"))
            if not all(current):
                raise ValueError(f"{path}: missing environment metadata")
            if environment is None:
                environment = current
            elif current != environment:
                raise ValueError(f"{path}: environment does not match the other reports")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(args.shapes, values, args), encoding="utf-8")
        latex_target.write_text(render_latex(args.shapes, values, args), encoding="utf-8")
        print(f"Table: {target}", flush=True)
        print(f"LaTeX: {latex_target}", flush=True)
        if any(value is None for value in values.values()):
            print("Some cases failed correctness and are marked INVALID; inspect their raw reports.", file=sys.stderr)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\nCompleted reports were preserved. Use --resume to continue.\n")


if __name__ == "__main__":
    main()
