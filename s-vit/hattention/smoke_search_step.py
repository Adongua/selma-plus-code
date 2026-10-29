"""Run one public-data alternating search step with a frozen ViT teacher.

Synthetic costs are allowed only when explicitly requested for this smoke;
this command does not export a structure or a trained checkpoint.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .group_costs import load_complete_tables
from .search import alternating_step, architecture_parameters, search_layers, weight_parameters
from .search_data import SearchDataset, collate_search, training_splits
from .student_checkpoint import file_sha256, load_student, make_checkpoint_search_student
from .teacher_guidance import load_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mpcvit-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--mapping", choices=("strict-id", "first-name"), default="first-name")
    parser.add_argument("--budget-online-seconds", type=float, default=9.0)
    costs = parser.add_mutually_exclusive_group(required=True)
    costs.add_argument("--cost-table", type=Path)
    costs.add_argument("--smoke-synthetic-costs", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if not 1 <= args.batch_size <= 16 or not 0 < args.budget_online_seconds < float("inf"):
        parser.error("invalid batch size or online budget")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    _, weight_items, arch_items, split = training_splits(args.dataset_root, seed=args.seed)
    source, checkpoint = load_student(args.checkpoint, args.mpcvit_root)
    student = make_checkpoint_search_student(source)
    del source
    guidance, teacher_metadata = load_teacher(args.teacher_dir, args.dataset_root, mapping=args.mapping)
    hidden = teacher_metadata["teacher_hidden_size"]
    student.search_feature_projection = nn.Linear(192, hidden)
    student = student.to(args.device).train()
    guidance = guidance.to(args.device).eval()
    layers = search_layers(student)
    heads = layers[0].config.num_heads
    if args.smoke_synthetic_costs:
        # A clearly marked local test fixture only; never claim measured cost.
        i = torch.arange(heads + 1, dtype=torch.float64)[:, None]
        j = torch.arange(heads + 1, dtype=torch.float64)[None, :]
        table = torch.where(i + j <= heads, 0.1 + i * 0.1 + (heads - i - j) * 0.01, float("nan"))
        cost_metadata = {"kind": "synthetic_smoke_only", "not_for_search_results": True}
    else:
        tables, metadata = load_complete_tables(args.cost_table)
        if metadata["shape_b_h_n_d"] != [args.batch_size, heads, 257, 16]:
            raise ValueError("cost-table shape must match (batch,12,257,16)")
        table = tables["online_seconds"]
        cost_metadata = {"kind": "measured", "table": str(args.cost_table.resolve()),
                         "table_sha256": file_sha256(args.cost_table), "metadata": metadata}
    weight_batch = collate_search([SearchDataset(args.dataset_root, weight_items)[i]
                                   for i in range(args.batch_size)])
    arch_batch = collate_search([SearchDataset(args.dataset_root, arch_items)[i]
                                 for i in range(args.batch_size)])
    def task(batch):
        inputs, images, labels = batch
        loss, _ = guidance.loss(student, images, inputs.to(args.device), labels.to(args.device))
        return loss
    weight_opt = torch.optim.AdamW(weight_parameters(student), lr=1e-5)
    arch_opt = torch.optim.Adam(architecture_parameters(student), lr=3e-3)
    before = [layer.structure_logits.detach().clone() for layer in layers]
    print(json.dumps({"event": "search_step_started", "device": args.device,
                      "cost_kind": cost_metadata["kind"]}), flush=True)
    metrics = alternating_step(student, weight_opt, arch_opt,
                               lambda: task(weight_batch), lambda: task(arch_batch),
                               {"online_seconds": [table] * len(layers)},
                               {"online_seconds": args.budget_online_seconds}, entropy_progress=0.0)
    delta = max(float((layer.structure_logits.detach() - original).abs().max())
                for layer, original in zip(layers, before))
    if not 0 < delta < float("inf"):
        raise ValueError("architecture logits did not update")
    report = {"status": "smoke_passed", "scope": "one public-data alternating step only",
              "checkpoint": checkpoint, "teacher": teacher_metadata, "split": split,
              "costs": cost_metadata, "budget_online_seconds": args.budget_online_seconds,
              "student": {"layers": len(layers), "heads_per_layer": heads,
                          "features": "student CLS 192 projected to teacher CLS feature",
                          "attention_weight_dropout_removed": True},
              "metrics": metrics, "max_architecture_logit_change": delta,
              "warning": "No trained checkpoint or selected architecture is produced; this is a smoke test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "cost_kind": cost_metadata["kind"], "metrics": metrics,
                      "max_architecture_logit_change": delta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
