"""Build the paper-style GeLU/LayerNorm tables from benchmark JSON reports.

Only actual non-warmup measurements are used. Missing methods, shapes, or
repetitions are errors rather than being filled with guessed values.
"""

import argparse
import json
import math
from pathlib import Path
import statistics


CONFIG = {
    "sgelu": {
        "shapes": ((1, 197), (1, 257), (1, 577), (8, 257)),
        "methods": (
            ("sgelu", "sGeLU (ours)", r"sGeLU (ours)"),
            ("erf", "CrypTen-erf (2021)", r"CrypTen-erf (2021)"),
            ("masked_fourier", "SecFormer masked Fourier (2024)*", r"SecFormer (2024)$^{*}$"),
            ("puma", "PUMA adapted (2025)", r"PUMA (2025)"),
            ("seaf_pwl_3pc", "SEAF adapted 3PC (2025)†", r"SEAF (2025)$^{\dagger}$"),
        ),
        "error": "error_vs_torch_gelu",
    },
    "slayernorm": {
        "shapes": ((1, 197, 256), (1, 257, 256), (1, 577, 256), (8, 257, 256)),
        "methods": (
            ("crypten_native", "CrypTen-native (2021)", r"CrypTen-native (2021)"),
            ("secformer", "SecFormer adapted (2024)", r"SecFormer (2024)"),
            ("puma", "PUMA adapted (2025)", r"PUMA (2025)"),
            ("sprint", "SPRINT adapted (2026)", r"SPRINT (2026)"),
            ("newton_engineering", "Newton engineering baseline", r"Newton engineering baseline"),
            ("slayernorm", "sLayerNorm (ours)", r"sLayerNorm (ours)"),
        ),
        "error": "error_vs_torch",
    },
}


def load_measurements(paths, operator, distribution, comm_metric):
    config = CONFIG[operator]
    samples = {}
    metadata = []
    seen_cases = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("world_size") != 3:
            raise ValueError(f"{path}: expected world_size=3")
        if report.get("status") == "benchmark_failed_correctness":
            raise ValueError(f"{path}: correctness gate failed")
        config_keys = ("mu",) if operator == "sgelu" else ("epsilon", "scale")
        metadata.append(tuple(report.get(key) for key in (
            "backend", "crypten_import", "crypten_default_precision_bits",
            "python", "torch", "repeats", "seed", *config_keys,
        )))
        per_case = {}
        for record in report["records"]:
            if record.get("warmup"):
                continue
            if operator == "slayernorm" and record.get("distribution") != distribution:
                continue
            shape = tuple(record.get("shape", report.get("shape", ())))
            for method, result in record["methods"].items():
                key = (shape, method)
                if result.get("correctness_gate", {}).get("passed") is False:
                    raise ValueError(f"{path}: {key} failed its correctness gate")
                online = result["online"]
                row = (float(online["seconds_max"]),
                       float(online[comm_metric]) / 1024.0,
                       float(max(online["rounds_per_rank"])),
                       float(result[config["error"]]["mae"]))
                if not all(math.isfinite(value) and value >= 0 for value in row):
                    raise ValueError(f"{path}: invalid measurement for {key}")
                per_case.setdefault(key, []).append(row)
        for key, rows in per_case.items():
            if key in seen_cases:
                raise ValueError(f"duplicate case {key}: supply one report per case")
            if len(rows) != report["repeats"]:
                raise ValueError(f"{path}: {key} has {len(rows)} samples, expected {report['repeats']}")
            seen_cases.add(key)
            samples[key] = tuple(statistics.mean(column) for column in zip(*rows))
    if len(set(metadata)) != 1:
        raise ValueError("reports use different backend, environment, precision or experiment settings")
    missing = [(shape, method) for shape in config["shapes"]
               for method, _, _ in config["methods"] if (shape, method) not in samples]
    if missing:
        raise ValueError(f"missing required shape/method cases: {missing}")
    return samples, metadata[0]


def format_row(values, bold=None, latex=False):
    cells = []
    for index, value in enumerate(values):
        text = (f"{value:.5f}" if index == 0 else
                f"{value:.1f}" if index == 1 else
                f"{value:.1f}" if index == 2 else f"{value:.5g}")
        if index == 2 and value.is_integer():
            text = str(int(value))
        if latex and index == 3 and value and (value < 0.001 or value >= 1000):
            mantissa, exponent = f"{value:.2e}".split("e")
            text = "$" + mantissa + r"\times 10^{" + str(int(exponent)) + "}$"
        if bold and index < 3 and bold[index]:
            text = r"\textbf{" + text + "}"
        cells.append(text)
    return cells


def render(operator, samples, metadata, distribution, comm_metric):
    config = CONFIG[operator]
    shapes, methods = config["shapes"], config["methods"]
    backend, crypten_path, precision = metadata[:3]
    header = ["Method"]
    for shape in shapes:
        header.extend([f"{shape} Time(s)", "Comm.(KiB)", "Rounds", "MAE"])
    rows = ["# " + ("sGeLU" if operator == "sgelu" else "sLayerNorm") + " microbenchmark",
            "", f"Backend: {backend}; world size: 3; CrypTen: {crypten_path}; default precision: Q{precision}.",
            f"Distribution: {distribution}; communication field: {comm_metric} / 1024 (KiB).",
            "One warmup excluded; measurements are means across reported repetitions. "
            "Time, communication and rounds refer to online fields; MAE is versus PyTorch. "
            "LaTeX boldface marks raw per-column minima, not an accuracy-qualified SOTA claim.",
            "", "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header)]
    for method, label, _ in methods:
        cells = [label]
        for shape in shapes:
            cells.extend(format_row(samples[(shape, method)]))
        rows.append("| " + " | ".join(cells) + " |")

    rows += ["", "## LaTeX", "", "```latex", r"% \usepackage{booktabs}",
             r"% \usepackage{graphicx}", r"\begin{table*}[t]", r"\centering",
             r"\caption{GPU online cost and numerical error on synthetic inputs.}",
             r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * (4 * len(shapes)) + "}",
             r"\toprule", r"\textbf{Method} " + " ".join(
                 r"& \multicolumn{4}{c}{\textbf{Shape $" + str(shape).replace(" ", "") + r"$}}"
                 for shape in shapes) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + i * 4, 5 + i * 4) for i in range(len(shapes))),
             " " + " ".join(r"& \textbf{Time (s)} & \textbf{Comm. (KiB)} & \textbf{Rounds} & \textbf{MAE}"
                            for _ in shapes) + r" \\", r"\midrule"]
    for method, _, label in methods:
        cells = [label]
        for shape in shapes:
            values = samples[(shape, method)]
            best = [min(samples[(shape, m)][i] for m, _, _ in methods) for i in range(3)]
            bold = [math.isclose(values[i], best[i], rel_tol=1e-12, abs_tol=1e-12) for i in range(3)]
            cells.extend(format_row(values, bold, latex=True))
        rows.append("  " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", "```", "",
             "The CrypTen provider work is included in its online time; Ring3 preprocessing "
             "is reported separately. " + ("Ring3 Gloo uses CPU staging, included in its phase time; "
             if backend == "gloo" else "Ring3 uses the selected backend; ") +
             "CrypTen uses its own Gloo/CUDA path. Collective byte counts are estimates. "
             "Ruler openings are not security validated. Check each method's accuracy before "
             "using its timing as a valid comparison."]
    if operator == "sgelu":
        rows += ["", "* SecFormer is a masked-Fourier engineering reference, not the full official protocol. "
                 "† SEAF is a CrypTen 3PC adaptation, not the original 2PC implementation."]
    else:
        rows += ["", "SecFormer, PUMA and SPRINT are same-framework 3PC adaptations; "
                 "Newton is an engineering baseline. SHAFT is omitted from this table."]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=CONFIG, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distribution", choices=("normal", "uniform"), default="normal")
    parser.add_argument("--comm-metric", choices=("system_bytes", "sent_bytes_sum"), default="system_bytes")
    args = parser.parse_args()
    try:
        samples, metadata = load_measurements(args.inputs, args.operator, args.distribution, args.comm_metric)
        content = render(args.operator, samples, metadata, args.distribution, args.comm_metric)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"cannot build table: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Table: {args.output}")


if __name__ == "__main__":
    main()
