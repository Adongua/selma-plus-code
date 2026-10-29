"""Validate a locally downloaded Hugging Face ViT on Tiny-ImageNet.

The default requires exact WordNet IDs. An explicitly selected first-name
mapping is an experimental hypothesis, not proof of equivalent WordNet senses.
Weights are loaded with weights_only.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


WNID = re.compile(r"n[0-9]{8}\Z")


def _validated_labels(wnids, id2label):
    wnids = sorted(wnids)
    if len(wnids) != 200 or len(set(wnids)) != 200 or any(not WNID.fullmatch(w) for w in wnids):
        raise ValueError("wnids.txt must contain 200 distinct WordNet IDs")
    labels = {int(k): str(v).strip() for k, v in id2label.items()}
    if set(labels) != set(range(200)):
        raise ValueError("teacher must provide exactly 200 indexed labels")
    if len(set(labels.values())) != 200:
        raise ValueError("teacher labels must be unique")
    return wnids, labels


def class_mapping(wnids, id2label):
    """Return teacher class indices in sorted student class order."""
    wnids, labels = _validated_labels(wnids, id2label)
    if len(set(labels.values())) != 200 or set(labels.values()) != set(wnids):
        raise ValueError("teacher id2label must identify the same 200 WordNet IDs; "
                         "do not guess a mapping from class names")
    inverse = {label: index for index, label in labels.items()}
    return wnids, [inverse[wnid] for wnid in wnids]


def read_first_names(words_file):
    """Read the first comma-separated name for each WordNet ID."""
    names = {}
    for line in words_file.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        wnid, description = line.split("\t", 1)
        name = description.split(",", 1)[0].strip().casefold()
        if wnid in names:
            raise ValueError(f"duplicate WordNet ID in words.txt: {wnid}")
        names[wnid] = name
    return names


def class_mapping_by_first_name(wnids, id2label, names):
    """Build a bijection by first name; this does not establish synset identity."""
    wnids, labels = _validated_labels(wnids, id2label)
    if any(not WNID.fullmatch(label) for label in labels.values()):
        raise ValueError("first-name mapping requires teacher WordNet IDs")
    missing = (set(wnids) | set(labels.values())) - set(names)
    if missing:
        raise ValueError(f"words.txt missing WordNet IDs: {sorted(missing)}")
    student_names = [names[wnid] for wnid in wnids]
    teacher_names = [names[labels[i]] for i in range(200)]
    if (any(not name for name in student_names + teacher_names)
            or len(set(student_names)) != 200 or len(set(teacher_names)) != 200):
        raise ValueError("first names must be nonempty and unique on both sides")
    if set(student_names) != set(teacher_names):
        raise ValueError("dataset and teacher first-name sets do not match")
    inverse = {name: index for index, name in enumerate(teacher_names)}
    indices = [inverse[name] for name in student_names]
    rows = [{"dataset_index": i, "dataset_wnid": wnid, "first_name": student_names[i],
             "teacher_index": indices[i], "teacher_wnid": labels[indices[i]],
             "different_wnid": wnid != labels[indices[i]]}
            for i, wnid in enumerate(wnids)]
    return wnids, indices, rows


def accuracy_summary(indices, support, correct1, correct5):
    count = sum(support[i] for i in indices)
    top1 = sum(correct1[i] for i in indices)
    top5 = sum(correct5[i] for i in indices)
    return {"samples": count, "classes_observed": sum(support[i] > 0 for i in indices),
            "correct_top1": top1, "correct_top5": top5,
            "top1": top1 / count if count else None,
            "top5": top5 / count if count else None}


def validation_items(root, wnids):
    val = root / "val"
    annotations = val / "val_annotations.txt"
    if not annotations.is_file():
        raise FileNotFoundError(f"validation annotations missing: {annotations}")

    # Support the original val/images layout and images reorganized under
    # val/<wnid>. Annotations remain the authority for names and labels.
    indices = {wnid: i for i, wnid in enumerate(wnids)}
    image_roots = [val / "images"] + [val / wnid for wnid in wnids]
    image_roots = [path for path in image_roots if path.is_dir()]
    if not image_roots:
        raise FileNotFoundError(f"no validation image directories found under {val}; "
                                "expected images/ or WordNet-ID class directories")
    image_paths = {}
    for image_root in image_roots:
        for path in image_root.rglob("*"):
            if path.is_file():
                image_paths.setdefault(path.name, []).append(path)

    result, seen = [], set()
    missing, ambiguous, misplaced = [], [], []
    for line_number, line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) < 2 or parts[1] not in indices:
            raise ValueError(f"invalid validation annotation at line {line_number}")
        name, wnid = parts[:2]
        if Path(name).name != name or name in seen:
            raise ValueError(f"invalid or duplicate image name at line {line_number}")
        seen.add(name)
        candidates = image_paths.get(name, [])
        if not candidates:
            missing.append(name)
            continue
        if len(candidates) > 1:
            ambiguous.append(name)
            continue
        path = candidates[0]
        class_dirs = [part for part in path.relative_to(val).parts[:-1] if part in indices]
        if any(folder != wnid for folder in class_dirs):
            misplaced.append((name, wnid, class_dirs))
            continue
        result.append((path, indices[wnid]))
    if ambiguous:
        raise ValueError("validation image filenames are ambiguous under val: "
                         f"{ambiguous[:5]} (and {max(0, len(ambiguous) - 5)} more)")
    if misplaced:
        raise ValueError("validation image class directory disagrees with annotation: "
                         f"{misplaced[:5]}")
    if missing:
        sample_files = sorted(image_paths)[:8]
        raise ValueError(
            f"{len(missing)} annotated validation images are missing under {val}; "
            f"first missing: {missing[:8]}; found {len(image_paths)} image files "
            f"with names such as {sample_files}. Check that validation images and "
            "val/val_annotations.txt come from the same Tiny-ImageNet archive."
        )
    if not result:
        raise ValueError("validation annotations are empty")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0 evaluates the full validation set; a positive value is a smoke test")
    parser.add_argument("--mapping", choices=("strict-id", "first-name"), default="strict-id",
                        help="first-name is an explicit, unverified label-mapping hypothesis")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_samples < 0:
        parser.error("batch size must be positive and max-samples nonnegative")
    if args.output.exists():
        parser.error("output already exists; choose a new output file")

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    teacher_dir = args.teacher_dir.resolve()
    config = AutoConfig.from_pretrained(teacher_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "vit":
        raise ValueError(f"expected a ViT model, got {config.model_type!r}")
    wnids = (args.dataset_root / "wnids.txt").read_text(encoding="utf-8").splitlines()
    words_file = args.dataset_root / "words.txt"
    if args.mapping == "first-name":
        names = read_first_names(words_file)
        classes, teacher_indices, mapping_rows = class_mapping_by_first_name(wnids, config.id2label, names)
    else:
        classes, teacher_indices = class_mapping(wnids, config.id2label)
        mapping_rows = [{"dataset_index": i, "dataset_wnid": wnid,
                         "teacher_index": teacher_indices[i], "teacher_wnid": wnid,
                         "different_wnid": False}
                        for i, wnid in enumerate(classes)]
    items = validation_items(args.dataset_root, classes)
    if args.max_samples:
        items = items[:args.max_samples]

    # Explicitly use PyTorch's restricted loader: never execute remote code.
    weight_file = teacher_dir / "pytorch_model.bin"
    if not weight_file.is_file():
        raise FileNotFoundError(f"expected {weight_file}")
    digest = hashlib.sha256()
    with weight_file.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    state = torch.load(weight_file, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected a plain Hugging Face tensor state_dict")
    model = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
    model.load_state_dict(state, strict=True)
    processor = AutoImageProcessor.from_pretrained(teacher_dir, local_files_only=True,
                                                   trust_remote_code=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(args.device).eval()
    permutation = torch.tensor(teacher_indices, dtype=torch.long, device=args.device)
    support, correct1, correct5 = Counter(), Counter(), Counter()
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            images = []
            for path, _ in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = {key: value.to(args.device) for key, value in
                      processor(images=images, return_tensors="pt").items()}
            logits = model(**inputs).logits
            if logits.ndim != 2 or logits.shape != (len(chunk), 200) or not torch.isfinite(logits).all():
                raise ValueError("teacher produced invalid logits")
            logits = logits.index_select(1, permutation)
            targets = torch.tensor([target for _, target in chunk], device=args.device)
            predictions = logits.topk(5, dim=1).indices
            top1_hits = (predictions[:, 0] == targets).tolist()
            top5_hits = (predictions == targets[:, None]).any(dim=1).tolist()
            for (_, target), hit1, hit5 in zip(chunk, top1_hits, top5_hits):
                support[target] += 1
                correct1[target] += int(hit1)
                correct5[target] += int(hit5)
            print(f"evaluated {min(start + len(chunk), len(items))}/{len(items)}", flush=True)
    overall = accuracy_summary(range(200), support, correct1, correct5)
    differing = [row["dataset_index"] for row in mapping_rows if row["different_wnid"]]
    matching = [row["dataset_index"] for row in mapping_rows if not row["different_wnid"]]
    per_class = [{**row, **accuracy_summary([row["dataset_index"]], support, correct1, correct5)}
                 for row in mapping_rows]
    report = {"status": "evaluated", "scope": "smoke_subset" if args.max_samples else "full_annotations",
              "teacher_dir": str(teacher_dir), "dataset_root": str(args.dataset_root.resolve()),
              "model_type": config.model_type, "architecture": config.architectures,
              "hidden_size": config.hidden_size, "layers": config.num_hidden_layers,
              "heads": config.num_attention_heads, "patch_size": config.patch_size,
              "image_size": config.image_size, "num_classes": len(classes),
              "class_order": "sorted Tiny-ImageNet WordNet IDs", "teacher_indices": teacher_indices,
              "mapping": {"method": args.mapping, "different_wnid_classes": len(differing),
                          "rows": mapping_rows,
                          "teacher_config_sha256": hashlib.sha256((teacher_dir / "config.json").read_bytes()).hexdigest(),
                          "dataset_wnids_sha256": hashlib.sha256((args.dataset_root / "wnids.txt").read_bytes()).hexdigest(),
                          "words_sha256": hashlib.sha256(words_file.read_bytes()).hexdigest()
                          if args.mapping == "first-name" else None,
                          "caveat": "First-name matching is an empirical hypothesis; identical names do not prove identical WordNet senses or training labels."
                          if args.mapping == "first-name" else None},
              "samples": len(items), "classes_observed": overall["classes_observed"],
              "top1": overall["top1"], "top5": overall["top5"],
              "groups": {"different_wnid": accuracy_summary(differing, support, correct1, correct5),
                         "same_wnid": accuracy_summary(matching, support, correct1, correct5)},
              "per_class": per_class,
              "device": args.device, "weights": str(weight_file), "weights_sha256": digest.hexdigest(),
              "security": "public pretrained candidate; restricted weights-only deserialization"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps({"report": str(args.output), "status": report["status"],
                      "scope": report["scope"], "mapping": args.mapping,
                      "different_wnid_classes": len(differing), "overall": overall,
                      "groups": report["groups"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
