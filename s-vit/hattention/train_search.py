"""Train and export a plaintext hAttention search model with measured costs.

This is architecture search, not secure inference or the final distillation run.
Checkpoints are written after each completed epoch; the official val split is
not used for optimization or model selection.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from .group_costs import load_complete_tables
from .search import (alternating_step, architecture_parameters, discretize,
                     export_fixed_student, search_layers,
                     temperature_at, weight_parameters)
from .search_data import SearchDataset, collate_search, training_splits
from .student_checkpoint import (configure_fixed_rsa_search, create_random_student, file_sha256, load_student,
                                 make_checkpoint_search_student)
from .teacher_guidance import load_teacher


def split_architecture_items(items, *, validation_per_class=10, seed=20261030):
    """Reserve a disjoint, class-balanced search holdout from train images."""
    by_class = {}
    for path, target in items:
        by_class.setdefault(target, []).append((path, target))
    if validation_per_class < 1 or not by_class:
        raise ValueError("validation_per_class must be positive and items nonempty")
    train, validation_by_class = [], []
    for target in sorted(by_class):
        values = sorted(by_class[target])
        if len(values) <= validation_per_class:
            raise ValueError(f"class {target} has too few architecture images")
        random.Random(f"holdout:{seed}:{target}").shuffle(values)
        validation_by_class.append(values[:validation_per_class])
        train.extend(values[validation_per_class:])
    # Interleave classes so a small validation subset is not one-class only.
    validation = [values[i] for i in range(validation_per_class)
                  for values in validation_by_class]
    def digest(rows):
        h = hashlib.sha256()
        for path, target in rows:
            h.update(f"{target}\t{path}\n".encode())
        return h.hexdigest()
    return train, validation, {"train_samples": len(train), "validation_samples": len(validation),
                               "validation_per_class": validation_per_class,
                               "train_sha256": digest(train), "validation_sha256": digest(validation)}


def loader(dataset, batch_size, *, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, collate_fn=collate_search, drop_last=False)


def train_epoch(model, teacher, weight_loader, architecture_loader, weight_optimizer,
                architecture_optimizer, table, budget, device, *, epoch, epochs,
                max_steps=0, max_rsa=108.0, lambda_cost=1.0, log_every=0,
                lambda_task=1.0, lambda_efficiency=0.0, task_loss_scale=1.0,
                architecture_warmup_epochs=0, fixed_temperature=False):
    model.train()
    layers = search_layers(model)
    temperature = 1.0 if fixed_temperature else temperature_at(epoch, epochs)
    for layer in layers:
        layer.temperature = temperature
    architecture_iterator = iter(architecture_loader)
    totals, steps = {}, 0
    for weight_batch in weight_loader:
        if max_steps and steps >= max_steps:
            break
        try:
            architecture_batch = next(architecture_iterator)
        except StopIteration:
            architecture_iterator = iter(architecture_loader)
            architecture_batch = next(architecture_iterator)

        task_components = {}

        def task(batch, stage):
            inputs, images, labels = batch
            loss, components = teacher.loss(model, images, inputs.to(device), labels.to(device))
            task_components[stage] = {key: float(value.detach()) for key, value in components.items()}
            return loss

        metrics = alternating_step(model, weight_optimizer, architecture_optimizer,
                                   lambda: task(weight_batch, "weight"),
                                   lambda: task(architecture_batch, "architecture"),
                                   {"online_seconds": [table] * len(layers)},
                                   {"online_seconds": budget}, max_rsa=max_rsa,
                                   lambda_cost=lambda_cost,
                                   lambda_task=lambda_task, lambda_efficiency=lambda_efficiency,
                                   task_loss_scale=task_loss_scale,
                                   entropy_progress=epoch / max(epochs - 1, 1),
                                   update_architecture=epoch >= architecture_warmup_epochs)
        if (getattr(model.classifier, "seq_pool", None) is False
                and all(layer.config.has_cls_token and layer.path_mode == "straight_through" for layer in layers)
                and all(bool((layer.structure_logits.argmax(dim=-1) == 2).all()) for layer in layers)):
            raise RuntimeError("architecture collapsed to all-Mixer: CLS cannot receive patch information")
        metrics["task_components"] = task_components
        for key in ("weight_loss", "architecture_loss"):
            totals[key] = totals.get(key, 0.0) + metrics[key]
        for key, value in metrics["architecture_components"].items():
            name = "architecture_" + key
            totals[name] = totals.get(name, 0.0) + value
        for stage, components in task_components.items():
            for key, value in components.items():
                name = f"{stage}_{key}"
                totals[name] = totals.get(name, 0.0) + value
        totals["expected_online_seconds"] = (totals.get("expected_online_seconds", 0.0)
                                               + metrics["expected_costs"]["online_seconds"])
        steps += 1
        if log_every and steps % log_every == 0:
            print(json.dumps({"event": "training_step", "epoch": epoch + 1,
                              "step": steps, **metrics}), flush=True)
    if not steps:
        raise ValueError("training epoch had no batches")
    return {"steps": steps, "temperature": temperature,
            "architecture_updates_enabled": epoch >= architecture_warmup_epochs,
            **{key: value / steps for key, value in totals.items()}}


def evaluate(model, dataset, device, batch_size, max_samples):
    """Classification metrics on a held-out train subset; no teacher involved."""
    if not 0 < max_samples <= len(dataset):
        raise ValueError("invalid number of validation samples")
    subset = torch.utils.data.Subset(dataset, range(max_samples))
    layers = search_layers(model)
    observations = [layer.observed_min_denominator.clone() for layer in layers]
    was_training = model.training
    model.eval()
    top1 = top5 = total = 0
    try:
        with torch.no_grad():
            for inputs, _, labels in loader(subset, batch_size, shuffle=False, seed=0):
                logits, _ = model(inputs.to(device))
                if logits.ndim != 2 or not torch.isfinite(logits).all():
                    raise ValueError("invalid validation logits")
                labels = labels.to(device)
                predictions = logits.topk(min(5, logits.size(1)), dim=1).indices
                top1 += int((predictions[:, 0] == labels).sum())
                top5 += int((predictions == labels[:, None]).any(dim=1).sum())
                total += labels.numel()
    finally:
        for layer, previous in zip(layers, observations):
            layer.observed_min_denominator.copy_(previous)
        model.train(was_training)
    return {"samples": total, "top1": top1 / total, "top5": top5 / total,
            "scope": "held-out training images, not official validation"}


def write_json(path, value, *, overwrite=False):
    if overwrite:
        temporary = path.with_suffix(path.suffix + ".partial")
        output_path = temporary
    else:
        output_path = path
    with output_path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
    if overwrite:
        output_path.replace(path)


def save_checkpoint(path, payload):
    """Write complete checkpoints atomically, refusing to replace existing runs."""
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"checkpoint already exists: {path} or {temporary}")
    try:
        with temporary.open("xb") as output:
            torch.save(payload, output)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def summarize_architecture(layers):
    """Summarize path probabilities without treating equal logits as RSA votes."""
    names = ("rsa", "lsa", "mixer")
    per_layer = []
    all_probabilities = []
    searchable_probabilities = []
    for layer in layers:
        p = layer.probabilities().detach().cpu()
        maxima = p.max(dim=-1, keepdim=True).values
        tied = torch.isclose(p, maxima, rtol=1e-6, atol=1e-8).sum(dim=-1) != 1
        choice = p.argmax(dim=-1)
        per_layer.append({"mean_probability": dict(zip(names, p.mean(dim=0).tolist())),
                          "unique_preferred_counts": {name: int(((choice == i) & ~tied).sum())
                                                      for i, name in enumerate(names)},
                          "tied_heads": int(tied.sum())})
        all_probabilities.append(p)
        if layer.fixed_rsa_positions:
            searchable_probabilities.append(p[layer.allowed_paths[:, 2].detach().cpu(), 2])
    p = torch.cat(all_probabilities, dim=0)
    summary = {"mean_probability": dict(zip(names, p.mean(dim=0).tolist())),
               "per_layer": per_layer}
    if searchable_probabilities:
        mixer = torch.cat(searchable_probabilities)
        summary["searchable_lsa_heads"] = int(mixer.numel())
        summary["mean_mixer_probability_on_searchable_heads"] = (
            float(mixer.mean()) if mixer.numel() else None)
    return summary


def selected_group_cost(layers, table):
    """Cost-table estimate for the argmax assignments actually used by hard mode."""
    total = 0.0
    for layer in layers:
        paths = layer.probabilities().detach().argmax(dim=-1)
        total += float(table[int((paths == 0).sum()), int((paths == 2).sum())])
    return total


def load_rsa_warmup_weights(model, path, classes, split, teacher_metadata=None):
    """Load a full-split fixed-RSA warmup; keep new search logits and observations."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != "hattention_rsa_warmup_v1":
        raise ValueError("expected an all-RSA warmup checkpoint")
    config = artifact.get("config", {})
    if config.get("overfit_diagnostic") is not False:
        raise ValueError("overfit diagnostic checkpoints cannot initialize a search")
    if config.get("class_order") != classes or config.get("training_split") != split:
        raise ValueError("warmup dataset split or class order differs from search")
    if config.get("loss") == "distill":
        if teacher_metadata is None or any(
                config.get("teacher_" + key) != teacher_metadata.get(key)
                for key in ("weights_sha256", "config_sha256", "processor_sha256")):
            raise ValueError("warmup teacher differs from the search teacher")
    state = artifact.get("model")
    if not isinstance(state, dict):
        raise ValueError("warmup checkpoint has no model state")
    own = model.state_dict()
    expected_missing = {key for key in own if key.endswith(".structure_logits")
                        or key.endswith(".observed_min_denominator")}
    if "search_feature_projection.weight" not in state:
        expected_missing |= {key for key in own if key.startswith("search_feature_projection.")}
    missing = set(own) - set(state)
    unexpected = set(state) - set(own)
    if missing != expected_missing or unexpected:
        raise ValueError(f"warmup model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    model.load_state_dict(state, strict=False)
    return artifact.get("metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mpcvit-root", "dataset-root", "teacher-dir", "cost-table", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="MPCViT weights, required for checkpoint initialization")
    parser.add_argument("--student-init", choices=("checkpoint", "scratch"), default="checkpoint")
    parser.add_argument("--warmup-checkpoint", type=Path,
                        help="full-split all-RSA warmup weights for scratch search")
    parser.add_argument("--resume", type=Path, help="resume a completed epoch checkpoint")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--budget-online-seconds", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 uses the entire weight split")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-per-class", type=int, default=10)
    parser.add_argument("--weight-lr", type=float)
    parser.add_argument("--architecture-lr", type=float, default=3e-3)
    parser.add_argument("--architecture-init", choices=("mpcvit", "uniform", "rsa"))
    parser.add_argument("--search-space", choices=("three_path", "fixed_rsa_lsa_mixer"), default="three_path",
                        help="freeze checkpoint RSA locations and search LSA/Mixer elsewhere")
    parser.add_argument("--mixer-init-probability", type=float, default=0.05)
    parser.add_argument("--architecture-warmup-epochs", type=int, default=0,
                        help="train model weights while architecture logits are frozen")
    parser.add_argument("--path-mode", choices=("soft", "straight_through"))
    parser.add_argument("--rsa-logit-margin", type=float, default=2.0)
    parser.add_argument("--lambda-cost", type=float)
    parser.add_argument("--lambda-task", type=float)
    parser.add_argument("--lambda-efficiency", type=float)
    parser.add_argument("--task-loss-scale", type=float,
                        help="normalizer for the teacher task loss in architecture optimization")
    parser.add_argument("--selection-cost-preference", type=float)
    parser.add_argument("--max-rsa", type=float, default=108.0, help="soft total RSA count limit across all layers")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--mapping", choices=("strict-id", "first-name"), default="first-name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    scratch = args.student_init == "scratch"
    args.architecture_init = args.architecture_init or ("rsa" if scratch else "mpcvit")
    args.path_mode = args.path_mode or ("straight_through" if scratch else "soft")
    args.weight_lr = args.weight_lr if args.weight_lr is not None else (3e-4 if scratch else 1e-5)
    args.lambda_cost = args.lambda_cost if args.lambda_cost is not None else (10.0 if scratch else 1.0)
    args.lambda_task = args.lambda_task if args.lambda_task is not None else 1.0
    args.lambda_efficiency = args.lambda_efficiency if args.lambda_efficiency is not None else 0.0
    args.task_loss_scale = args.task_loss_scale if args.task_loss_scale is not None else (50.0 if scratch else 1.0)
    args.selection_cost_preference = (args.selection_cost_preference if args.selection_cost_preference is not None
                                      else 0.0)
    if scratch and (args.checkpoint is not None or args.architecture_init == "mpcvit"):
        parser.error("scratch initialization cannot use a checkpoint or MPCViT architecture prior")
    if not scratch and args.checkpoint is None:
        parser.error("checkpoint initialization requires --checkpoint")
    constrained = args.search_space == "fixed_rsa_lsa_mixer"
    if constrained and (scratch or args.architecture_init != "mpcvit" or args.path_mode != "soft"):
        parser.error("fixed RSA search requires checkpoint initialization, mpcvit prior and soft paths")
    if args.warmup_checkpoint is not None and (not scratch or args.architecture_init != "rsa"):
        parser.error("warmup weights require scratch initialization and the RSA architecture prior")
    if (args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or
        args.eval_samples < 1 or args.max_steps_per_epoch < 0 or args.log_every < 0 or
        not 0 <= args.architecture_warmup_epochs < args.epochs or
        not math.isfinite(args.mixer_init_probability) or not 0 < args.mixer_init_probability < 0.5 or
        any(not math.isfinite(x) or x <= 0 for x in
            (args.budget_online_seconds, args.weight_lr, args.architecture_lr, args.max_rsa)) or
        any(not math.isfinite(x) or x < 0 for x in (args.lambda_cost, args.lambda_task,
            args.lambda_efficiency, args.selection_cost_preference)) or
        not math.isfinite(args.task_loss_scale) or args.task_loss_scale <= 0 or
        not math.isfinite(args.rsa_logit_margin) or args.rsa_logit_margin <= 0):
        parser.error("invalid training, budget, or learning-rate arguments")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    root = args.output_root.resolve()
    if args.resume:
        if not root.is_dir():
            parser.error("resume requires the existing output directory")
        if args.resume.resolve().parent != root:
            parser.error("resume checkpoint must belong to output-root")
    else:
        root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    classes, weight_items, architecture_items, split = training_splits(args.dataset_root, seed=args.seed)
    arch_train, arch_val, holdout = split_architecture_items(
        architecture_items, validation_per_class=args.validation_per_class, seed=args.seed)
    if args.eval_samples > len(arch_val):
        parser.error("eval-samples exceeds the held-out search split")
    source, checkpoint_metadata = (create_random_student(args.mpcvit_root) if scratch else
                                   load_student(args.checkpoint, args.mpcvit_root))
    model = make_checkpoint_search_student(
        source, prior_strength=1.0 if args.architecture_init == "mpcvit" else 0.0)
    initial_paths = (configure_fixed_rsa_search(model, source, mixer_probability=args.mixer_init_probability)
                     if constrained else None)
    for layer in search_layers(model):
        layer.path_mode = args.path_mode
        if args.architecture_init == "rsa":
            with torch.no_grad():
                layer.structure_logits.zero_()
                layer.structure_logits[:, 0] = args.rsa_logit_margin
    del source
    teacher, teacher_metadata = load_teacher(args.teacher_dir, args.dataset_root, mapping=args.mapping)
    model.search_feature_projection = nn.Linear(192, teacher_metadata["teacher_hidden_size"])
    warmup_metrics = (load_rsa_warmup_weights(model, args.warmup_checkpoint, classes, split,
                                              teacher_metadata=teacher_metadata)
                      if args.warmup_checkpoint is not None else None)
    device = torch.device(args.device)
    model, teacher = model.to(device), teacher.to(device).eval()
    layers = search_layers(model)
    heads = layers[0].config.num_heads
    tables, cost_metadata = load_complete_tables(args.cost_table)
    # Search minibatches need not match the independently measured deployment batch.
    if cost_metadata["shape_b_h_n_d"] != [1, heads, 257, 16]:
        raise ValueError("cost table must describe deployment shape (1,12,257,16)")
    table = tables["online_seconds"]
    if constrained:
        minimum_possible = sum(min(float(table[paths.count("rsa"), mixer_count])
                                   for mixer_count in range(paths.count("lsa") + 1))
                               for paths in initial_paths)
        if args.budget_online_seconds + 1e-12 < minimum_possible:
            raise ValueError(f"budget {args.budget_online_seconds} is below the measured-table "
                             f"minimum {minimum_possible:.6f} for fixed RSA positions")
    budgets = {"online_seconds": args.budget_online_seconds}
    costs = {"online_seconds": [table] * len(layers)}
    config = {"epochs": args.epochs, "budget_online_seconds": args.budget_online_seconds,
              "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
              "eval_samples": args.eval_samples, "max_steps_per_epoch": args.max_steps_per_epoch,
              "validation_per_class": args.validation_per_class, "weight_lr": args.weight_lr,
              "architecture_lr": args.architecture_lr, "lambda_cost": args.lambda_cost,
              "architecture_init": args.architecture_init,
              "search_space": args.search_space,
              "mixer_init_probability": args.mixer_init_probability,
              "architecture_warmup_epochs": args.architecture_warmup_epochs,
              "student_init": args.student_init, "path_mode": args.path_mode,
              "rsa_logit_margin": args.rsa_logit_margin, "lambda_task": args.lambda_task,
              "lambda_efficiency": args.lambda_efficiency,
              "task_loss_scale": args.task_loss_scale,
              "selection_cost_preference": args.selection_cost_preference,
              "max_rsa": args.max_rsa, "seed": args.seed, "mapping": args.mapping,
              "device": args.device, "cost_table_sha256": file_sha256(args.cost_table),
              "checkpoint_sha256": checkpoint_metadata["sha256"],
              "warmup_checkpoint_sha256": (file_sha256(args.warmup_checkpoint)
                                           if args.warmup_checkpoint is not None else None),
              "teacher_weights_sha256": teacher_metadata["weights_sha256"],
              "teacher_config_sha256": teacher_metadata["config_sha256"],
              "teacher_processor_sha256": teacher_metadata["processor_sha256"],
              "training_split": split, "search_holdout": holdout}
    weight_opt = torch.optim.AdamW(weight_parameters(model), lr=args.weight_lr)
    arch_opt = torch.optim.Adam(architecture_parameters(model), lr=args.architecture_lr)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        saved_config = dict(checkpoint.get("config", {}))
        # Runs predating this option always used the MPCViT prior.
        saved_config.setdefault("architecture_init", "mpcvit")
        saved_config.setdefault("warmup_checkpoint_sha256", None)
        saved_config.setdefault("search_space", "three_path")
        saved_config.setdefault("mixer_init_probability", 0.05)
        saved_config.setdefault("architecture_warmup_epochs", 0)
        for key, default in (("student_init", "checkpoint"), ("path_mode", "soft"),
                             ("rsa_logit_margin", 2.0), ("lambda_task", 1.0),
                             ("lambda_efficiency", 0.0), ("task_loss_scale", 1.0),
                             ("selection_cost_preference", 0.0)):
            saved_config.setdefault(key, default)
        if checkpoint.get("format") != "hattention_search_checkpoint_v1" or saved_config != config:
            raise ValueError("resume checkpoint configuration or data/model hashes differ")
        start = checkpoint["completed_epochs"]
        if not 0 < start <= args.epochs:
            raise ValueError("invalid completed epoch count")
        if args.resume.resolve().name != f"epoch-{start:04d}.pt":
            raise ValueError("resume checkpoint filename and completed epoch disagree")
        model.load_state_dict(checkpoint["model"], strict=True)
        weight_opt.load_state_dict(checkpoint["weight_optimizer"])
        arch_opt.load_state_dict(checkpoint["architecture_optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
    else:
        write_json(root / "run.json", {"config": config, "checkpoint": checkpoint_metadata,
                   "teacher": teacher_metadata, "warmup_metrics": warmup_metrics,
                   "cost_table": str(args.cost_table.resolve()),
                   "cost_metadata": cost_metadata, "class_order": classes,
                   "initial_fixed_paths": initial_paths,
                   "initial_architecture": summarize_architecture(layers),
                   "note": "Search is plaintext. Official validation is reserved for final evaluation."})
    weight_data = SearchDataset(args.dataset_root, weight_items)
    arch_data = SearchDataset(args.dataset_root, arch_train)
    val_data = SearchDataset(args.dataset_root, arch_val)
    if constrained and not args.resume:
        baseline = export_fixed_student(model, initial_paths)
        baseline_metrics = evaluate(baseline, val_data, device, args.eval_batch_size, args.eval_samples)
        baseline_cost = sum(float(table[paths.count("rsa"), 0]) for paths in initial_paths)
        write_json(root / "baseline.json", {"validation_fixed": baseline_metrics,
                   "estimated_online_seconds": baseline_cost, "table_minimum_seconds": minimum_possible,
                   "note": "Held-out training images may have been seen by the pretrained checkpoint."})
        print(json.dumps({"event": "baseline", "validation_fixed": baseline_metrics,
                          "estimated_online_seconds": baseline_cost,
                          "table_minimum_seconds": minimum_possible}), flush=True)
        del baseline
    for epoch in range(start, args.epochs):
        print(json.dumps({"event": "epoch_started", "epoch": epoch + 1, "epochs": args.epochs}), flush=True)
        metrics = train_epoch(model, teacher,
                              loader(weight_data, args.batch_size, shuffle=True, seed=args.seed + 2 * epoch),
                              loader(arch_data, args.batch_size, shuffle=True, seed=args.seed + 2 * epoch + 1),
                              weight_opt, arch_opt, table, args.budget_online_seconds, device,
                              epoch=epoch, epochs=args.epochs, max_steps=args.max_steps_per_epoch,
                              max_rsa=args.max_rsa, lambda_cost=args.lambda_cost,
                              lambda_task=args.lambda_task, lambda_efficiency=args.lambda_efficiency,
                              task_loss_scale=args.task_loss_scale,
                              architecture_warmup_epochs=args.architecture_warmup_epochs,
                              fixed_temperature=constrained,
                              log_every=args.log_every)
        metrics["validation_mixed"] = evaluate(model, val_data, device, args.eval_batch_size, args.eval_samples)
        metrics["architecture"] = summarize_architecture(layers)
        metrics["argmax_structure_online_seconds"] = selected_group_cost(layers, table)
        metrics["epoch"] = epoch + 1
        payload = {"format": "hattention_search_checkpoint_v1", "config": config,
                   "completed_epochs": epoch + 1, "model": model.state_dict(),
                   "weight_optimizer": weight_opt.state_dict(), "architecture_optimizer": arch_opt.state_dict(),
                   "cpu_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
        checkpoint_path = root / f"epoch-{epoch + 1:04d}.pt"
        save_checkpoint(checkpoint_path, payload)
        write_json(root / f"epoch-{epoch + 1:04d}.json", metrics)
        print(json.dumps({"event": "epoch_complete", "checkpoint": str(checkpoint_path), **metrics}), flush=True)
    # Temperature is an ordinary attribute, so restore the final schedule on resume.
    for layer in layers:
        layer.temperature = 1.0 if constrained else temperature_at(args.epochs - 1, args.epochs)
    selection = discretize(layers, costs, budgets, cost_preference=args.selection_cost_preference)
    if constrained:
        selection["fixed_rsa_observations"] = [
            {"layer": index, "heads_unobserved_or_below_threshold": [
                head for head, (allowed, observed) in enumerate(zip(
                    layer.allowed_paths[:, 0].detach().cpu().tolist(),
                    layer.observed_min_denominator.detach().cpu().tolist()))
                if allowed and (not math.isfinite(observed) or observed < layer.denominator_min)],
             "threshold": layer.denominator_min}
            for index, layer in enumerate(layers)]
        selection["fixed_rsa_warning"] = ("RSA positions are retained even when observations fail; "
                                          "these observations do not establish secure numeric validity.")
    fixed = export_fixed_student(model, selection["head_types"])
    selection["validation_fixed"] = evaluate(fixed, val_data, device, args.eval_batch_size, args.eval_samples)
    selection["source_checkpoint"] = str((root / f"epoch-{args.epochs:04d}.pt").resolve())
    selection["cost_table_sha256"] = config["cost_table_sha256"]
    selection["note"] = ("Estimated attention core cost only; search holdout metrics are not official validation. "
                         "RSA range observations and Ruler are not a security proof. "
                         "Budget is enforced on the fixed cost estimate, not an accuracy guarantee.")
    save_checkpoint(root / "fixed-student.pt", {"format": "hattention_fixed_student_v1",
                    "state_dict": fixed.state_dict(), "head_types": selection["head_types"],
                    "student_init": args.student_init,
                    "source_checkpoint_sha256": checkpoint_metadata["sha256"]})
    write_json(root / "selection.json", selection)
    print(json.dumps({"event": "search_complete", "selection": str(root / "selection.json"),
                      "fixed_student": str(root / "fixed-student.pt"),
                      "costs": selection["costs"], "validation_fixed": selection["validation_fixed"]}), flush=True)


if __name__ == "__main__":
    main()
