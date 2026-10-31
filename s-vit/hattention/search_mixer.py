"""Greedy fixed-head Mixer search from a trained MPCViT checkpoint.

All training and selection are plaintext. Costs are estimates from a measured
grouped Ring3 table; they are not end-to-end measurements or security proofs.
"""

import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from .config import HAttentionConfig
from .group_costs import load_complete_tables
from .search_data import SearchDataset, training_splits
from .student_checkpoint import file_sha256, load_student
from .teacher_guidance import load_teacher
from .train_search import evaluate, loader, save_checkpoint, split_architecture_items, write_json
from .vit_integration import make_fixed_student


def balanced_items(items, count, seed):
    """Interleave randomly selected images per class to avoid sorted-class bias."""
    groups = {}
    for item in items:
        groups.setdefault(item[1], []).append(item)
    if not groups or count < 1 or count > len(items):
        raise ValueError("invalid balanced sample count")
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    rows = []
    for index in range(max(map(len, groups.values()))):
        for label in sorted(groups):
            if index < len(groups[label]):
                rows.append(groups[label][index])
                if len(rows) == count:
                    return rows
    raise ValueError("could not construct balanced sample")


def head_paths(model):
    return [list(block.self_attn.config.head_types) for block in model.classifier.blocks]


def count_paths(paths):
    return dict(Counter(path for layer in paths for path in layer))


def measured_cost(paths, table):
    total = 0.0
    for heads in paths:
        value = float(table[heads.count("rsa"), heads.count("mixer")])
        if not math.isfinite(value) or value < 0:
            raise ValueError("missing or invalid measured group cost")
        total += value
    return total


def score(model, dataset, device, batch_size, scope):
    metrics = evaluate(model, dataset, device, batch_size, len(dataset))
    return {**metrics, "scope": scope}


def replace_head(model, layer_index, head_index, path):
    attention = model.classifier.blocks[layer_index].self_attn
    paths = list(attention.config.head_types)
    paths[head_index] = path
    attention.config = replace(attention.config, head_types=tuple(paths))


def load_mixer_search_model(artifact_path, checkpoint_path, mpcvit_root, *, device="cpu"):
    """Reload a fixed candidate with its exact source checkpoint and paths."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != "hattention_mixer_search_v1":
        raise ValueError("expected a Mixer search artifact")
    source, metadata = load_student(checkpoint_path, mpcvit_root)
    if artifact.get("source_checkpoint_sha256") != metadata["sha256"]:
        raise ValueError("source checkpoint hash differs from search artifact")
    paths = artifact.get("head_types")
    blocks = source.classifier.blocks
    if not isinstance(paths, (tuple, list)) or len(paths) != len(blocks):
        raise ValueError("one path configuration is required per layer")
    configurations = [HAttentionConfig(block.self_attn.num_heads, tuple(heads), patch_grid=(16, 16))
                      for block, heads in zip(blocks, paths)]
    model = make_fixed_student(source, configurations, remove_attention_dropout=True)
    state = artifact["state_dict"]
    projection = state.get("search_feature_projection.weight")
    if projection is not None:
        if projection.ndim != 2 or projection.shape[1] != 192:
            raise ValueError("invalid feature projection in search artifact")
        model.search_feature_projection = nn.Linear(192, projection.shape[0])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), artifact


def candidates(paths, table, maximum_cost, minimum_non_mixer):
    """List one-head substitutions that preserve the global hard constraints."""
    non_mixer = sum(p != "mixer" for layer in paths for p in layer)
    if non_mixer <= minimum_non_mixer:
        return []
    rows = []
    for layer_index, heads in enumerate(paths):
        for head_index, previous in enumerate(heads):
            if previous == "mixer":
                continue
            proposed = [list(layer) for layer in paths]
            proposed[layer_index][head_index] = "mixer"
            seconds = measured_cost(proposed, table)
            if seconds <= maximum_cost + 1e-12:
                rows.append({"layer": layer_index, "head": head_index,
                             "previous": previous, "estimated_online_seconds": seconds})
    return rows


def recover(model, teacher, dataset, device, *, steps, batch_size, lr, seed,
            lambda_kd, lambda_feature):
    if not steps:
        return {"steps": 0, "mean_loss": None}
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses, cycle = [], 0
    while len(losses) < steps:
        for inputs, images, labels in loader(dataset, batch_size, shuffle=True, seed=seed + cycle):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if teacher is None:
                logits, _ = model(inputs)
                loss = F.cross_entropy(logits, labels)
            else:
                loss, _ = teacher.loss(model, images, inputs, labels,
                                       lambda_kd=lambda_kd, lambda_feature=lambda_feature)
            if not torch.isfinite(loss):
                raise ValueError("non-finite recovery loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) == steps:
                break
        cycle += 1
    return {"steps": len(losses), "mean_loss": sum(losses) / len(losses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "mpcvit-root", "dataset-root", "cost-table", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="continue a completed run stopped by --max-stages")
    parser.add_argument("--recovery-loss", choices=("ce", "distill"), default="ce")
    parser.add_argument("--mapping", choices=("strict-id", "first-name"), default="first-name")
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-feature", type=float, default=1.0)
    parser.add_argument("--max-top1-drop", type=float, default=0.05,
                        help="maximum absolute Top-1 drop from the original model on selection data")
    parser.add_argument("--min-top1", type=float, default=0.0,
                        help="optional absolute minimum selection Top-1")
    parser.add_argument("--max-cost-increase", type=float, default=0.0,
                        help="allowed seconds above original estimated attention-core cost")
    parser.add_argument("--budget-online-seconds", type=float,
                        help="optional final cost goal; intermediate stages may exceed it")
    parser.add_argument("--target-mixers", type=int, default=1)
    parser.add_argument("--minimum-non-mixer", type=int, default=1)
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=0,
                        help="0 checks every eligible head; positive values select a seeded subset")
    parser.add_argument("--sensitivity-samples", type=int, default=200)
    parser.add_argument("--selection-samples", type=int, default=1000)
    parser.add_argument("--holdout-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recovery-steps", type=int, default=20)
    parser.add_argument("--recovery-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (min(args.target_mixers, args.minimum_non_mixer, args.max_stages, args.top_k,
            args.sensitivity_samples, args.selection_samples, args.holdout_samples, args.batch_size) < 1
            or args.candidate_limit < 0 or args.recovery_steps < 0
            or not math.isfinite(args.max_top1_drop) or not 0 <= args.max_top1_drop <= 1
            or not math.isfinite(args.min_top1) or not 0 <= args.min_top1 <= 1
            or not math.isfinite(args.max_cost_increase) or args.max_cost_increase < 0
            or (args.budget_online_seconds is not None and
                (not math.isfinite(args.budget_online_seconds) or args.budget_online_seconds <= 0))
            or not math.isfinite(args.recovery_lr) or args.recovery_lr <= 0
            or any(not math.isfinite(x) or x < 0 for x in (args.lambda_kd, args.lambda_feature))):
        parser.error("invalid search parameters")
    if args.recovery_loss == "distill" and args.teacher_dir is None:
        parser.error("--recovery-loss distill requires --teacher-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if root.exists() and not args.resume:
        parser.error("output-root already exists; choose a new directory")
    if args.resume and not root.is_dir():
        parser.error("resume requires an existing output-root")

    torch.manual_seed(args.seed)
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=args.seed)
    arch_items, heldout_items, holdout_meta = split_architecture_items(architecture_items, seed=args.seed)
    if args.sensitivity_samples > args.selection_samples:
        parser.error("sensitivity-samples must not exceed selection-samples")
    selection_items = balanced_items(arch_items, args.selection_samples, args.seed)
    if args.holdout_samples > len(heldout_items):
        parser.error("holdout-samples exceeds held-out split")
    sensitivity_data = SearchDataset(args.dataset_root, selection_items[:args.sensitivity_samples])
    selection_data = SearchDataset(args.dataset_root, selection_items)
    holdout_data = SearchDataset(args.dataset_root, heldout_items[:args.holdout_samples])
    weight_data = SearchDataset(args.dataset_root, weight_items)
    source, source_metadata = load_student(args.checkpoint, args.mpcvit_root)
    configurations = []
    for block in source.classifier.blocks:
        attention = block.self_attn
        alpha = attention.alpha.detach().reshape(-1).tolist()
        configurations.append(HAttentionConfig(attention.num_heads,
                              tuple("rsa" if value == 1 else "lsa" for value in alpha),
                              patch_grid=(16, 16)))
    model = make_fixed_student(source, configurations, remove_attention_dropout=True)
    tables, metadata = load_complete_tables(args.cost_table)
    if metadata.get("shape_b_h_n_d") != [1, 12, 257, 16] or metadata.get("world_size") != 3:
        raise ValueError("cost table must match the three-party 1x12x257x16 attention core")
    table = tables["online_seconds"]
    teacher = teacher_metadata = None
    if args.recovery_loss == "distill":
        teacher, teacher_metadata = load_teacher(args.teacher_dir, args.dataset_root, mapping=args.mapping)
        model.search_feature_projection = nn.Linear(192, teacher_metadata["teacher_hidden_size"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    if teacher is not None:
        teacher = teacher.to(device).eval()
    baseline_paths = head_paths(model)
    baseline_cost = measured_cost(baseline_paths, table)
    maximum_cost = baseline_cost + args.max_cost_increase
    settings = {"seed": args.seed, "batch_size": args.batch_size, "top_k": args.top_k,
                "candidate_limit": args.candidate_limit, "sensitivity_samples": args.sensitivity_samples,
                "selection_samples": args.selection_samples, "holdout_samples": args.holdout_samples,
                "max_top1_drop": args.max_top1_drop, "min_top1": args.min_top1,
                "max_cost_increase": args.max_cost_increase,
                "minimum_non_mixer": args.minimum_non_mixer, "recovery_loss": args.recovery_loss,
                "recovery_steps": args.recovery_steps, "recovery_lr": args.recovery_lr,
                "lambda_kd": args.lambda_kd, "lambda_feature": args.lambda_feature,
                "device": args.device, "mapping": args.mapping}
    cost_hash = file_sha256(args.cost_table)
    if args.resume:
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if (report.get("status") != "completed_plaintext_not_security_validated"
                or report.get("termination") != "max_stages_reached"):
            raise ValueError("resume requires a completed run stopped at max-stages")
        if (report.get("settings") != settings or report.get("dataset_split") != split
                or report.get("holdout_split") != holdout_meta or report.get("teacher") != teacher_metadata
                or report.get("source", {}).get("sha256") != source_metadata["sha256"]
                or report.get("cost_table_sha256") != cost_hash):
            raise ValueError("resume settings, data, teacher, checkpoint or cost table differ")
        first_stage = len(report["stages"]) + 1
        if args.max_stages < first_stage:
            raise ValueError("resume requires a larger total --max-stages")
        model, _ = load_mixer_search_model(root / report["final_artifact"], args.checkpoint,
                                            args.mpcvit_root, device=device)
        floor = report["selection_top1_floor"]
        maximum_cost = report["maximum_estimated_online_seconds"]
        report["status"] = "running"
        report["target_mixers"] = args.target_mixers
        report["budget_online_seconds"] = args.budget_online_seconds
        write_json(root / "report.json", report, overwrite=True)
        print(json.dumps({"event": "resume", "next_stage": first_stage,
                          "floor": floor, "maximum_estimated_online_seconds": maximum_cost}), flush=True)
    else:
        first_stage = 1
        baseline_selection = score(model, selection_data, device, args.batch_size, "adaptive selection split")
        baseline_holdout = score(model, holdout_data, device, args.batch_size, "held-out training images")
        floor = max(args.min_top1, baseline_selection["top1"] - args.max_top1_drop)
        root.mkdir(parents=True, exist_ok=False)
        report = {"status": "running", "scope": "plaintext search; secure inference not validated",
              "source": source_metadata, "cost_table_sha256": file_sha256(args.cost_table),
              "cost_metadata": metadata, "dataset_split": split, "holdout_split": holdout_meta,
              "teacher": teacher_metadata, "class_order": classes, "settings": settings,
              "selection_samples": len(selection_data), "sensitivity_samples": len(sensitivity_data),
              "holdout_samples": len(holdout_data), "max_top1_drop": args.max_top1_drop,
              "selection_top1_floor": floor, "maximum_estimated_online_seconds": maximum_cost,
              "target_mixers": args.target_mixers, "budget_online_seconds": args.budget_online_seconds,
              "minimum_non_mixer": args.minimum_non_mixer,
              "recovery": {"loss": args.recovery_loss, "steps": args.recovery_steps,
                           "lr": args.recovery_lr, "lambda_kd": args.lambda_kd,
                           "lambda_feature": args.lambda_feature},
              "baseline": {"head_types": baseline_paths, "path_counts": count_paths(baseline_paths),
                           "estimated_online_seconds": baseline_cost, "selection": baseline_selection,
                           "holdout": baseline_holdout}, "stages": [],
              "notes": "Selection is adaptive on training-split images; holdout is not used by the selection algorithm, "
                       "but is displayed after each completed run. "
                       "The original pretrained checkpoint may have seen these holdout training images. "
                       "Cost is an estimated sum of grouped attention-core measurements, not full-model latency. "
                       "RSA ranges and Ruler privacy are not validated. Head screening and greedy search are not global."}
        write_json(root / "report.json", report)
        print(json.dumps({"event": "baseline", "selection": baseline_selection, "holdout": baseline_holdout,
                          "floor": floor, "estimated_online_seconds": baseline_cost}), flush=True)
    termination = "max_stages_reached"
    for stage in range(first_stage, args.max_stages + 1):
        current_paths = head_paths(model)
        rows = candidates(current_paths, table, maximum_cost, args.minimum_non_mixer)
        if args.candidate_limit and len(rows) > args.candidate_limit:
            rows = random.Random(args.seed + stage).sample(rows, args.candidate_limit)
        print(json.dumps({"event": "stage_started", "stage": stage, "candidates": len(rows)}), flush=True)
        if not rows:
            termination = "no_cost_feasible_or_connected_candidate"
            break
        for index, row in enumerate(rows, 1):
            replace_head(model, row["layer"], row["head"], "mixer")
            try:
                row["sensitivity"] = score(model, sensitivity_data, device, args.batch_size,
                                            "sensitivity subset of selection split")
            finally:
                replace_head(model, row["layer"], row["head"], row["previous"])
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({"event": "sensitivity_progress", "stage": stage,
                                  "completed": index, "total": len(rows)}), flush=True)
        rows.sort(key=lambda row: (-row["sensitivity"]["top1"],
                                   row["estimated_online_seconds"], row["layer"], row["head"]))
        finalists = rows[:args.top_k]
        best_model, best_row = None, None
        for row in finalists:
            candidate = copy.deepcopy(model)
            replace_head(candidate, row["layer"], row["head"], "mixer")
            before = score(candidate, selection_data, device, args.batch_size, "adaptive selection split")
            row["before_recovery"] = before
            before_state = copy.deepcopy(candidate.state_dict()) if args.recovery_steps else None
            row["recovery"] = recover(candidate, teacher, weight_data, device, steps=args.recovery_steps,
                                      batch_size=args.batch_size, lr=args.recovery_lr,
                                      seed=args.seed + stage, lambda_kd=args.lambda_kd,
                                      lambda_feature=args.lambda_feature)
            after = score(candidate, selection_data, device, args.batch_size, "adaptive selection split")
            row["after_recovery"] = after
            if before_state is not None and before["top1"] > after["top1"]:
                candidate.load_state_dict(before_state)
                row["selected_version"] = "before_recovery"
                row["selection"] = before
            else:
                row["selected_version"] = "after_recovery" if args.recovery_steps else "without_recovery"
                row["selection"] = after
            row["accepted_by_accuracy"] = row["selection"]["top1"] >= floor
            if row["accepted_by_accuracy"] and (best_row is None or
                    (row["estimated_online_seconds"], -row["selection"]["top1"], row["layer"], row["head"]) <
                    (best_row["estimated_online_seconds"], -best_row["selection"]["top1"], best_row["layer"], best_row["head"])):
                best_model, best_row = candidate, row
            print(json.dumps({"event": "candidate_evaluated", "stage": stage,
                              "layer": row["layer"], "head": row["head"],
                              "selection_top1": row["selection"]["top1"],
                              "accepted_by_accuracy": row["accepted_by_accuracy"]}), flush=True)
        stage_report = {"stage": stage, "screened": len(rows), "candidates": rows,
                        "selected": best_row}
        report["stages"].append(stage_report)
        write_json(root / f"stage-{stage:04d}.json", stage_report)
        if best_model is None:
            termination = "no_shortlisted_candidate_meets_accuracy_floor"
            write_json(root / "report.json", report, overwrite=True)
            break
        model = best_model
        current = head_paths(model)
        save_checkpoint(root / f"stage-{stage:04d}.pt", {
            "format": "hattention_mixer_search_v1", "state_dict": model.state_dict(),
            "head_types": current, "source_checkpoint_sha256": source_metadata["sha256"],
            "selection": best_row["selection"], "class_order": classes,
            "cost_table_sha256": report["cost_table_sha256"]})
        write_json(root / "report.json", report, overwrite=True)
        print(json.dumps({"event": "stage_complete", "stage": stage,
                          "selected": {"layer": best_row["layer"], "head": best_row["head"]},
                          "path_counts": count_paths(current),
                          "estimated_online_seconds": measured_cost(current, table),
                          "selection_top1": best_row["selection"]["top1"]}), flush=True)
    final_paths = head_paths(model)
    final_selection = score(model, selection_data, device, args.batch_size, "adaptive selection split")
    final_holdout = score(model, holdout_data, device, args.batch_size, "held-out training images")
    mixers = count_paths(final_paths).get("mixer", 0)
    final_cost = measured_cost(final_paths, table)
    report.update({"status": "completed_plaintext_not_security_validated", "termination": termination,
                   "target_mixers_reached": mixers >= args.target_mixers,
                   "budget_reached": None if args.budget_online_seconds is None else
                                     final_cost <= args.budget_online_seconds,
                   "final": {"head_types": final_paths, "path_counts": count_paths(final_paths),
                             "estimated_online_seconds": final_cost,
                             "selection": final_selection, "holdout": final_holdout}})
    final_artifact = "final.pt" if not args.resume else f"final-resume-{first_stage:04d}.pt"
    report["final_artifact"] = final_artifact
    save_checkpoint(root / final_artifact, {"format": "hattention_mixer_search_v1",
                    "state_dict": model.state_dict(), "head_types": final_paths,
                    "source_checkpoint_sha256": source_metadata["sha256"], "class_order": classes,
                    "cost_table_sha256": report["cost_table_sha256"]})
    write_json(root / "report.json", report, overwrite=True)
    print(json.dumps({"event": "search_complete", "report": str(root / "report.json"),
                      "termination": termination, "target_mixers_reached": mixers >= args.target_mixers,
                      "budget_reached": report["budget_reached"],
                      "final": report["final"]}), flush=True)


if __name__ == "__main__":
    main()
