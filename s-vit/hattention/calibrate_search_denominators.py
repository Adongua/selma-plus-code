"""Recalibrate final mixed-model RSA denominators and compare fixed selections.

Calibration and evaluation are disjoint held-out training images. This is a
plaintext diagnostic, not a secure range check or a formal safety guarantee.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Subset

from .group_costs import load_complete_tables
from .search import discretize, export_fixed_student, search_layers, temperature_at
from .search_data import SearchDataset, training_splits
from .student_checkpoint import file_sha256, load_student, make_checkpoint_search_student
from .train_search import evaluate, loader, save_checkpoint, split_architecture_items, write_json


def collect_denominators(model, dataset, device, batch_size):
    """Collect row denominators per head, restoring all historical observations."""
    layers = search_layers(model)
    history = [layer.observed_min_denominator.clone() for layer in layers]
    chunks = [[] for _ in layers]
    was_training = model.training
    for i, layer in enumerate(layers):
        if hasattr(layer, "denominator_observer"):
            raise ValueError("denominator observer already installed")
        def observe(value, index=i):
            # (batch, head, token) -> (head, all batch/token rows)
            chunks[index].append(value.transpose(0, 1).reshape(value.shape[1], -1).cpu())
        layer.denominator_observer = observe
    model.eval()
    try:
        with torch.no_grad():
            for inputs, _, _ in loader(dataset, batch_size, shuffle=False, seed=0):
                model(inputs.to(device))
    finally:
        for layer, old in zip(layers, history):
            del layer.denominator_observer
            layer.observed_min_denominator.copy_(old)
        model.train(was_training)
    if not all(chunks):
        raise ValueError("empty calibration or missing layer observations")
    return [torch.cat(parts, dim=1) for parts in chunks]


def summarize(values, threshold):
    """Report finite row distributions; any nonfinite sample disallows RSA."""
    rows = []
    for head in values:
        valid = head[torch.isfinite(head)].float()
        if not valid.numel():
            rows.append({"rows": head.numel(), "nonfinite": head.numel(), "min": None,
                         "p01": None, "p05": None, "median": None,
                         "fraction_below_threshold": None})
            continue
        quantiles = torch.quantile(valid, torch.tensor([0.01, 0.05, 0.5]))
        rows.append({"rows": head.numel(), "nonfinite": int(head.numel() - valid.numel()),
                     "min": float(valid.min()), "p01": float(quantiles[0]),
                     "p05": float(quantiles[1]), "median": float(quantiles[2]),
                     "fraction_below_threshold": float((valid < threshold).sum() / head.numel())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("search-checkpoint", "checkpoint", "mpcvit-root", "dataset-root", "cost-table", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if min(args.calibration_samples, args.eval_samples, args.batch_size) < 1:
        parser.error("sample and batch counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output-root already exists; choose a new directory")
    checkpoint = torch.load(args.search_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "hattention_search_checkpoint_v1":
        raise ValueError("expected an hAttention search checkpoint")
    config = checkpoint["config"]
    if checkpoint["completed_epochs"] != config["epochs"]:
        raise ValueError("expected the final completed epoch checkpoint")
    if file_sha256(args.checkpoint) != config["checkpoint_sha256"] or file_sha256(args.cost_table) != config["cost_table_sha256"]:
        raise ValueError("source checkpoint or cost-table hash differs from search")
    _, _, architecture_items, split = training_splits(args.dataset_root, seed=config["seed"])
    _, heldout, holdout = split_architecture_items(architecture_items,
        validation_per_class=config["validation_per_class"], seed=config["seed"])
    if split != config["training_split"] or holdout != config["search_holdout"]:
        raise ValueError("dataset split differs from checkpoint")
    n_cal, n_eval = args.calibration_samples, args.eval_samples
    if n_cal + n_eval > len(heldout):
        raise ValueError("calibration plus evaluation samples exceed held-out split")
    dataset = SearchDataset(args.dataset_root, heldout)
    cal = Subset(dataset, range(n_cal))
    evaluation = Subset(dataset, range(n_cal, n_cal + n_eval))
    source, source_metadata = load_student(args.checkpoint, args.mpcvit_root)
    model = make_checkpoint_search_student(source)
    weight = checkpoint["model"]["search_feature_projection.weight"]
    model.search_feature_projection = nn.Linear(weight.shape[1], weight.shape[0])
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    layers = search_layers(model)
    for layer in layers:
        layer.temperature = temperature_at(config["epochs"] - 1, config["epochs"])
    tables, metadata = load_complete_tables(args.cost_table)
    if metadata["shape_b_h_n_d"] != [1, layers[0].config.num_heads, 257, 16]:
        raise ValueError("cost table has incompatible deployment shape")
    costs = {"online_seconds": [tables["online_seconds"]] * len(layers)}
    budgets = {"online_seconds": config["budget_online_seconds"]}
    historical = discretize(layers, costs, budgets)
    print(json.dumps({"event": "calibration_started", "samples": n_cal}), flush=True)
    observations = collect_denominators(model, cal, device, args.batch_size)
    summaries = [summarize(x, layer.denominator_min) for x, layer in zip(observations, layers)]
    historical_min = [layer.observed_min_denominator.clone() for layer in layers]
    try:
        for layer, values in zip(layers, observations):
            # Fail closed for NaN/Inf; discretize rejects nonfinite observations.
            minima = values.min(dim=1).values
            minima[~torch.isfinite(values).all(dim=1)] = float("nan")
            layer.observed_min_denominator.copy_(minima.to(layer.observed_min_denominator.device))
        calibrated = discretize(layers, costs, budgets)
    finally:
        for layer, old in zip(layers, historical_min):
            layer.observed_min_denominator.copy_(old)
    print(json.dumps({"event": "evaluation_started", "samples": n_eval}), flush=True)
    mixed_eval = evaluate(model, evaluation, device, args.batch_size, n_eval)
    results = {}
    fixed_models = {}
    for name, selection in (("historical", historical), ("calibrated", calibrated)):
        fixed = export_fixed_student(model, selection["head_types"])
        results[name] = {"selection": selection, "path_counts": dict(Counter(
            path for layer in selection["head_types"] for path in layer)),
            "validation": evaluate(fixed, evaluation, device, args.batch_size, n_eval)}
        fixed_models[name] = fixed
    for index, summary in enumerate(summaries):
        for head, row in enumerate(summary):
            old = historical_min[index][head]
            row["historical_min"] = float(old) if torch.isfinite(old) else None
            row["historical_path"] = historical["head_types"][index][head]
            row["calibrated_path"] = calibrated["head_types"][index][head]
    changed = sum(a != b for old, new in zip(historical["head_types"], calibrated["head_types"])
                  for a, b in zip(old, new))
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "plaintext_diagnostic_not_security_validated",
              "search_checkpoint_sha256": file_sha256(args.search_checkpoint),
              "architecture_init": config.get("architecture_init", "mpcvit"),
              "cost_table_sha256": config["cost_table_sha256"],
              "calibration_samples": n_cal, "evaluation_samples": n_eval,
              "split": "disjoint subsets of search-held-out training images; official validation unused",
              "caveat": "Mixed-model plaintext observations do not establish fixed-model or 3PC safety; "
                        "selection uses a calibration set previously used for search monitoring.",
              "threshold": layers[0].denominator_min, "layers": summaries,
              "mixed_validation": mixed_eval, "changed_heads": changed, "comparisons": results}
    write_json(root / "report.json", report)
    for name in results:
        save_checkpoint(root / f"fixed-student-{name}.pt", {
            "format": "hattention_fixed_student_v1", "state_dict": fixed_models[name].state_dict(),
            "head_types": results[name]["selection"]["head_types"],
            "source_checkpoint_sha256": source_metadata["sha256"]})
    print(json.dumps({"report": str(root / "report.json"), "changed_heads": changed, "mixed": mixed_eval,
                      "historical": results["historical"]["validation"],
                      "calibrated": results["calibrated"]["validation"],
                      "costs": {k: v["selection"]["costs"] for k, v in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
