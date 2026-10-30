"""Checkpoint compatibility, disjoint data splits and teacher loss checks."""

from pathlib import Path
import tempfile
import types
import unittest

from PIL import Image
import torch
from torch import nn

from hattention.config import HAttentionConfig
from hattention.search import (alternating_step, architecture_parameters, make_search_student,
                               search_layers, weight_parameters)
from hattention.search_data import SearchDataset, training_splits
from hattention.student_checkpoint import load_fixed_student, load_student, make_checkpoint_search_student
from hattention.search import export_fixed_student
from hattention.teacher_guidance import TeacherGuidance


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv, self.proj = nn.Linear(8, 24), nn.Linear(8, 8)
        self.attn_drop, self.proj_drop = nn.Identity(), nn.Identity()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, x):
        return self.self_attn(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        self.classifier.blocks = nn.ModuleList([Block()])
        self.fc = nn.Linear(8, 200)

    def forward(self, inputs):
        hidden = self.classifier.blocks[0](inputs)
        return self.fc(hidden[:, 0]), [hidden]


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Linear(6, 200)

    def forward(self, pixel_values, output_hidden_states, return_dict):
        hidden = pixel_values[:, None, :]
        return types.SimpleNamespace(logits=self.logits(pixel_values), hidden_states=[hidden])


class Processor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.stack(images)}


class PipelineTests(unittest.TestCase):
    def test_balanced_deterministic_disjoint_split_and_image_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [f"n{i:08d}" for i in range(200)]
            (root / "wnids.txt").write_text("\n".join(reversed(ids)))
            for wnid in ids:
                directory = root / "train" / wnid / "images"
                directory.mkdir(parents=True)
                for number in range(2):
                    Image.new("RGB", (4, 4), (number, 10, 20)).save(directory / f"{number}.JPEG")
            first = training_splits(root, weight_per_class=1, architecture_per_class=1, seed=4)
            second = training_splits(root, weight_per_class=1, architecture_per_class=1, seed=4)
            self.assertEqual(first, second)
            classes, weight, arch, metadata = first
            self.assertEqual(classes, ids)
            self.assertEqual((len(weight), len(arch)), (200, 200))
            self.assertFalse(set(x[0] for x in weight) & set(x[0] for x in arch))
            self.assertFalse(metadata["validation_used_for_search"])
            pixels, rgb, target = SearchDataset(root, weight)[0]
            self.assertEqual(tuple(pixels.shape), (3, 64, 64))
            self.assertEqual(rgb.mode, "RGB")
            self.assertEqual(target, 0)

    def test_teacher_mapping_frozen_and_alternating_step_updates_student(self):
        torch.manual_seed(1)
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "lsa"), patch_grid=(2, 2))])
        model.search_feature_projection = nn.Linear(8, 6)
        teacher = TeacherGuidance(Teacher(), Processor(), list(reversed(range(200))))
        images, inputs, labels = [torch.randn(6)], torch.randn(1, 5, 8), torch.tensor([4])
        aligned, _ = teacher.targets(images, torch.device("cpu"))
        with torch.no_grad():
            original = teacher.teacher.logits(images[0].unsqueeze(0))
        self.assertTrue(torch.equal(aligned, original.flip(-1)))
        before = model.search_feature_projection.weight.detach().clone()
        weight_opt = torch.optim.AdamW(weight_parameters(model), lr=1e-3)
        arch_opt = torch.optim.Adam(architecture_parameters(model), lr=1e-2)
        task = lambda: teacher.loss(model, images, inputs, labels)[0]
        table = torch.tensor([[1., 0.5, 0.], [3., 2., float("nan")], [5., float("nan"), float("nan")]])
        result = alternating_step(model, weight_opt, arch_opt, task, task, {"time": [table]}, {"time": 2.0})
        self.assertTrue(torch.isfinite(torch.tensor(result["architecture_loss"])))
        self.assertFalse(torch.equal(before, model.search_feature_projection.weight))
        self.assertTrue(all(param.grad is None and not param.requires_grad for param in teacher.parameters()))

    def test_real_checkpoint_load_and_prior_when_available(self):
        repo = Path(__file__).resolve().parents[3]
        checkpoint = repo / "mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.5-kd.pth.tar"
        if not checkpoint.is_file():
            self.skipTest("local MPCViT checkpoint unavailable")
        source, metadata = load_student(checkpoint, repo / "mpcvit")
        self.assertEqual(metadata["active_tensors_loaded"], 116)
        self.assertEqual(len(metadata["ignored_legacy_keys"]), 148)
        student = make_checkpoint_search_student(source)
        self.assertEqual(len(search_layers(student)), 9)
        for block, layer in zip(source.classifier.blocks, search_layers(student)):
            alpha = block.self_attn.alpha.reshape(-1)
            self.assertTrue(torch.equal(layer.structure_logits[:, 0].detach(), alpha))
            self.assertTrue(torch.equal(layer.structure_logits[:, 1].detach(), 1 - alpha))
        student.search_feature_projection = nn.Linear(192, 6)
        paths = [["mixer"] * 12 for _ in range(9)]
        fixed = export_fixed_student(student, paths)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "fixed.pt"
            torch.save({"format": "hattention_fixed_student_v1", "state_dict": fixed.state_dict(),
                        "head_types": paths, "source_checkpoint_sha256": metadata["sha256"]}, artifact)
            restored, info = load_fixed_student(artifact, checkpoint, repo / "mpcvit")
            self.assertEqual(info["head_types"], paths)
            self.assertEqual(restored.classifier.blocks[0].self_attn.config.head_types, tuple(paths[0]))
            self.assertTrue(torch.equal(restored.classifier.blocks[0].self_attn.qkv.weight,
                                        fixed.classifier.blocks[0].self_attn.qkv.weight))


if __name__ == "__main__":
    unittest.main()
