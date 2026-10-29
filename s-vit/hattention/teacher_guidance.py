"""Teacher logit alignment and CLS feature distillation for public-data search."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .student_checkpoint import file_sha256
from .validate_teacher import class_mapping, class_mapping_by_first_name, read_first_names


class TeacherGuidance(nn.Module):
    """Frozen Hugging Face teacher; model inputs use its own image processor."""

    def __init__(self, teacher, processor, permutation, *, teacher_temperature=2.0):
        super().__init__()
        if teacher_temperature <= 0:
            raise ValueError("teacher temperature must be positive")
        self.teacher = teacher.requires_grad_(False).eval()
        self.processor = processor
        self.register_buffer("permutation", torch.tensor(permutation, dtype=torch.long))
        self.teacher_temperature = float(teacher_temperature)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def targets(self, images, device):
        inputs = {key: value.to(device) for key, value in
                  self.processor(images=images, return_tensors="pt").items()}
        with torch.no_grad():
            output = self.teacher(**inputs, output_hidden_states=True, return_dict=True)
            logits = output.logits.index_select(1, self.permutation.to(device))
            features = output.hidden_states[-1][:, 0, :]
        if logits.ndim != 2 or logits.shape[1] != 200 or not torch.isfinite(logits).all():
            raise ValueError("teacher produced invalid logits")
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("teacher produced invalid CLS features")
        return logits, features

    def loss(self, student, images, student_inputs, labels, *, lambda_kd=1.0, lambda_feature=1.0):
        student_logits, student_features = student(student_inputs)
        teacher_logits, teacher_features = self.targets(images, student_inputs.device)
        projection = getattr(student, "search_feature_projection", None)
        if not isinstance(projection, nn.Linear):
            raise ValueError("student must have a trainable search_feature_projection")
        projected = projection(student_features[-1][:, 0, :])
        if projected.shape != teacher_features.shape or student_logits.shape != teacher_logits.shape:
            raise ValueError("teacher/student output dimensions do not match")
        temperature = self.teacher_temperature
        ce = F.cross_entropy(student_logits, labels)
        kd = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                      F.softmax(teacher_logits / temperature, dim=-1), reduction="batchmean") * temperature ** 2
        feature = F.mse_loss(projected, teacher_features)
        return ce + lambda_kd * kd + lambda_feature * feature, {"ce": ce, "kd": kd, "feature": feature}


def load_teacher(teacher_dir, dataset_root, *, mapping="first-name", temperature=2.0):
    """Load the pinned local teacher with restricted deserialization and explicit mapping."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

    teacher_dir, root = Path(teacher_dir).resolve(), Path(dataset_root).resolve()
    config = AutoConfig.from_pretrained(teacher_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != "vit":
        raise ValueError("expected a ViT teacher")
    wnids = (root / "wnids.txt").read_text(encoding="utf-8").splitlines()
    if mapping == "first-name":
        classes, indices, rows = class_mapping_by_first_name(
            wnids, config.id2label, read_first_names(root / "words.txt"))
    elif mapping == "strict-id":
        classes, indices = class_mapping(wnids, config.id2label)
        rows = []
    else:
        raise ValueError("mapping must be strict-id or first-name")
    weight_file = teacher_dir / "pytorch_model.bin"
    state = torch.load(weight_file, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("teacher state_dict must contain only tensors")
    teacher = AutoModelForImageClassification.from_config(config, trust_remote_code=False)
    teacher.load_state_dict(state, strict=True)
    processor = AutoImageProcessor.from_pretrained(teacher_dir, local_files_only=True,
                                                   trust_remote_code=False, use_fast=False)
    guidance = TeacherGuidance(teacher, processor, indices, teacher_temperature=temperature)
    metadata = {"teacher_dir": str(teacher_dir), "weights_sha256": file_sha256(weight_file),
                "config_sha256": file_sha256(teacher_dir / "config.json"),
                "processor_sha256": file_sha256(teacher_dir / "preprocessor_config.json"),
                "mapping": mapping, "teacher_indices": indices, "class_order": classes,
                "different_wnid_classes": sum(row["different_wnid"] for row in rows),
                "teacher_hidden_size": config.hidden_size,
                "mapping_warning": "first-name is an unverified label-mapping hypothesis" if rows else None}
    return guidance, metadata
