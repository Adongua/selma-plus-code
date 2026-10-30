"""Search runner regression tests using tiny public synthetic inputs."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from hattention.config import HAttentionConfig
from hattention.search import (architecture_parameters, make_search_student, search_layers,
                               weight_parameters)
from hattention.teacher_guidance import TeacherGuidance
from hattention.train_search import (evaluate, loader, save_checkpoint,
                                    split_architecture_items, train_epoch)
from hattention import train_search
from hattention.student_checkpoint import make_checkpoint_search_student
from hattention.reference import rsa_attention
from hattention.search import discretize, search_objective


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

    def forward(self, value):
        return self.self_attn(value)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        self.classifier.blocks = nn.ModuleList([Block()])
        self.fc = nn.Linear(8, 200)

    def forward(self, value):
        hidden = self.classifier.blocks[0](value)
        return self.fc(hidden[:, 0]), [hidden]


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 200)

    def forward(self, pixel_values, output_hidden_states, return_dict):
        return types.SimpleNamespace(logits=self.fc(pixel_values),
                                     hidden_states=[pixel_values[:, None, :]])


class Processor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": torch.stack(images)}


class Fixture(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.randn(5, 8, generator=generator) * .2, torch.randn(6, generator=generator), index


class WideStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        block = nn.Module()
        attention = nn.Module()
        attention.num_heads = 12
        attention.qkv, attention.proj = nn.Linear(192, 576), nn.Linear(192, 192)
        attention.attn_drop, attention.proj_drop = nn.Identity(), nn.Identity()
        block.self_attn = attention
        self.classifier.blocks = nn.ModuleList([block])
        self.fc = nn.Linear(192, 200)

    def forward(self, value):
        hidden = self.classifier.blocks[0].self_attn(value)
        return self.fc(hidden[:, 0]), [hidden]


class WideFixture(Dataset):
    def __init__(self, root, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return (torch.randn(5, 192, generator=generator) * .1,
                torch.randn(6, generator=generator), self.items[index][1])


class SearchRunnerTests(unittest.TestCase):
    def test_straight_through_starts_with_rsa_and_has_search_gradients(self):
        torch.manual_seed(4)
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))])
        layer = search_layers(model)[0]
        layer.path_mode = "straight_through"
        with torch.no_grad():
            layer.structure_logits[:, 0] = 2.0
        value = torch.randn(1, 5, 8) * .2
        q, k, v = layer.qkv(value).reshape(1, 5, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        expected = rsa_attention(q, k, v, epsilon=layer.config.epsilon,
                                 radius=layer.config.srelu_radius,
                                 score_scale=layer.config.resolved_rho(layer.head_dim))
        expected = layer.proj_drop(layer.proj(expected.transpose(1, 2).reshape(1, 5, 8)))
        self.assertTrue(torch.allclose(layer(value), expected, atol=1e-7))
        grad = torch.autograd.grad(layer(value).square().sum(), layer.structure_logits)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_efficiency_objective_and_selection_reward_lower_cost(self):
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))])
        layer = search_layers(model)[0]
        layer.observed_min_denominator.fill_(1)
        model(torch.randn(1, 5, 8) * .1)
        table = torch.tensor([[2., 1., 0.], [4., 3., float("nan")], [6., float("nan"), float("nan")]])
        loss, parts = search_objective([layer], torch.tensor(0.), {"time": [table]}, {"time": 10.},
                                       lambda_task=0., lambda_cost=0., lambda_efficiency=1.,
                                       lambda_num=0., lambda_count=0., lambda_entropy=0.)
        grad = torch.autograd.grad(loss, layer.structure_logits)[0]
        self.assertGreater(float(grad[:, 0].mean()), float(grad[:, 2].mean()))
        self.assertGreater(float(parts["efficiency"]), 0)
        with torch.no_grad():
            layer.structure_logits[:, 1] = 1.
        likely = discretize([layer], {"time": [table]}, {"time": 10.})
        efficient = discretize([layer], {"time": [table]}, {"time": 10.}, cost_preference=20.)
        self.assertEqual(likely["head_types"], [["lsa", "lsa"]])
        self.assertEqual(efficient["head_types"], [["mixer", "mixer"]])

    def test_uniform_prior_preserves_weights_and_equalizes_probabilities(self):
        torch.manual_seed(5)
        source = Student()
        source.classifier.blocks[0].self_attn.alpha = torch.tensor([1., 0.])
        prior = make_checkpoint_search_student(source, prior_strength=1)
        uniform = make_checkpoint_search_student(source, prior_strength=0)
        a, b = search_layers(prior)[0], search_layers(uniform)[0]
        self.assertTrue(torch.equal(a.qkv.weight, b.qkv.weight))
        self.assertTrue(torch.allclose(b.probabilities(), torch.full((2, 3), 1 / 3)))
        self.assertTrue(torch.equal(a.structure_logits.detach(), torch.tensor([[1., 0., 0.], [0., 1., 0.]])))
        summary = train_search.summarize_architecture([b])
        self.assertEqual(summary["per_layer"][0]["tied_heads"], 2)
        self.assertEqual(summary["per_layer"][0]["unique_preferred_counts"],
                         {"rsa": 0, "lsa": 0, "mixer": 0})

    def test_balanced_holdout_is_disjoint_deterministic_and_interleaved(self):
        items = [(f"{c}/{i}", c) for c in range(3) for i in range(5)]
        train, val, metadata = split_architecture_items(items, validation_per_class=2, seed=1)
        self.assertEqual((len(train), len(val)), (9, 6))
        self.assertEqual([target for _, target in val[:3]], [0, 1, 2])
        self.assertFalse(set(train) & set(val))
        self.assertEqual((train, val, metadata), split_architecture_items(items, validation_per_class=2, seed=1))

    def test_training_validation_and_checkpoint_resume(self):
        torch.manual_seed(1)
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "lsa"), patch_grid=(2, 2))])
        model.search_feature_projection = nn.Linear(8, 6)
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        weights = torch.optim.AdamW(weight_parameters(model), lr=1e-3)
        architecture = torch.optim.Adam(architecture_parameters(model), lr=1e-2)
        table = torch.tensor([[1., .5, 0.], [3., 2., float("nan")], [5., float("nan"), float("nan")]])
        data = Fixture()
        before = search_layers(model)[0].structure_logits.detach().clone()
        metrics = train_epoch(model, teacher, loader(data, 1, shuffle=True, seed=1),
                              loader(data, 1, shuffle=True, seed=2), weights, architecture,
                              table, 2., torch.device("cpu"), epoch=0, epochs=2, max_steps=2,
                              max_rsa=2.)
        self.assertEqual(metrics["steps"], 2)
        self.assertFalse(torch.equal(before, search_layers(model)[0].structure_logits))
        observed = search_layers(model)[0].observed_min_denominator.clone()
        result = evaluate(model, data, torch.device("cpu"), 2, 4)
        self.assertEqual(result["samples"], 4)
        self.assertTrue(torch.equal(observed, search_layers(model)[0].observed_min_denominator))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch-0001.pt"
            save_checkpoint(path, {"model": model.state_dict(), "opt": weights.state_dict(),
                                   "architecture": architecture.state_dict()})
            saved = torch.load(path, weights_only=True)
            def second_epoch():
                return train_epoch(model, teacher, loader(data, 1, shuffle=True, seed=3),
                                   loader(data, 1, shuffle=True, seed=4), weights, architecture,
                                   table, 2., torch.device("cpu"), epoch=1, epochs=2, max_steps=2,
                                   max_rsa=2.)
            second_epoch()
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            model.load_state_dict(saved["model"])
            weights.load_state_dict(saved["opt"])
            architecture.load_state_dict(saved["architecture"])
            second_epoch()
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, {})

    def test_entrypoint_recovers_after_interrupted_export(self):
        architecture_items = [(f"{c}/{i}", c) for c in range(2) for i in range(2)]
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        teacher_meta = {"teacher_hidden_size": 6, "weights_sha256": "w",
                        "config_sha256": "c", "processor_sha256": "p"}
        def make_model(source, **kwargs):
            return make_search_student(WideStudent(), [HAttentionConfig(12, ("lsa",) * 12,
                                                                         patch_grid=(2, 2))])
        table = torch.zeros(13, 13, dtype=torch.float64)
        for i in range(13):
            for j in range(13):
                if i + j > 12:
                    table[i, j] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost = root / "cost.json"
            cost.write_text("{}")
            output = root / "output"
            argv = ["train_search", "--checkpoint", str(root / "model"), "--mpcvit-root", str(root),
                    "--dataset-root", str(root), "--teacher-dir", str(root), "--cost-table", str(cost),
                    "--output-root", str(output), "--epochs", "1", "--budget-online-seconds", "9",
                    "--validation-per-class", "1", "--eval-samples", "2", "--max-steps-per-epoch", "1",
                    "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch.object(train_search, "training_splits", return_value=(
                     ["class0", "class1"], [("w0", 0), ("w1", 1)], architecture_items, {"seed": 1})), \
                 patch.object(train_search, "load_student", return_value=(None, {"sha256": "student"})), \
                 patch.object(train_search, "make_checkpoint_search_student", side_effect=make_model), \
                 patch.object(train_search, "load_teacher", return_value=(teacher, teacher_meta)), \
                 patch.object(train_search, "load_complete_tables", return_value=(
                     {"online_seconds": table}, {"shape_b_h_n_d": [1, 12, 257, 16]})), \
                 patch.object(train_search, "SearchDataset", WideFixture), redirect_stdout(io.StringIO()):
                with patch.object(train_search, "discretize", side_effect=RuntimeError("interrupted export")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted export"):
                        train_search.main()
                checkpoint = output / "epoch-0001.pt"
                self.assertTrue(checkpoint.exists())
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["config"]["architecture_init"], "mpcvit")
                saved["config"].pop("architecture_init")
                torch.save(saved, checkpoint)  # emulate a legacy checkpoint
                with patch("sys.argv", argv + ["--architecture-init", "uniform", "--resume", str(checkpoint)]):
                    with self.assertRaisesRegex(ValueError, "configuration"):
                        train_search.main()
                with patch("sys.argv", argv + ["--resume", str(checkpoint)]):
                    train_search.main()
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(len(selection["head_types"]), 1)
            self.assertTrue((output / "fixed-student.pt").exists())

    def test_scratch_entrypoint_uses_no_checkpoint_and_exports(self):
        items = [(f"{c}/{i}", c) for c in range(2) for i in range(2)]
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        meta = {"teacher_hidden_size": 6, "weights_sha256": "w", "config_sha256": "c",
                "processor_sha256": "p"}
        table = torch.zeros(13, 13, dtype=torch.float64)
        def make_model(source, **kwargs):
            return make_search_student(WideStudent(), [HAttentionConfig(12, ("lsa",) * 12,
                                                                         patch_grid=(2, 2))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost = root / "cost.json"
            cost.write_text("{}")
            output = root / "output"
            argv = ["train_search", "--student-init", "scratch", "--mpcvit-root", str(root),
                    "--dataset-root", str(root), "--teacher-dir", str(root), "--cost-table", str(cost),
                    "--output-root", str(output), "--epochs", "1", "--budget-online-seconds", "9",
                    "--validation-per-class", "1", "--eval-samples", "2", "--max-steps-per-epoch", "1",
                    "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch.object(train_search, "training_splits", return_value=(
                     ["class0", "class1"], [("w0", 0), ("w1", 1)], items, {"seed": 1})), \
                 patch.object(train_search, "create_random_student", return_value=(None, {"sha256": None})), \
                 patch.object(train_search, "load_student", side_effect=AssertionError("loaded checkpoint")), \
                 patch.object(train_search, "make_checkpoint_search_student", side_effect=make_model), \
                 patch.object(train_search, "load_teacher", return_value=(teacher, meta)), \
                 patch.object(train_search, "load_complete_tables", return_value=(
                     {"online_seconds": table}, {"shape_b_h_n_d": [1, 12, 257, 16]})), \
                 patch.object(train_search, "SearchDataset", WideFixture), redirect_stdout(io.StringIO()):
                with patch.object(train_search, "discretize", side_effect=RuntimeError("interrupted export")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted export"):
                        train_search.main()
                with patch("sys.argv", argv + ["--resume", str(output / "epoch-0001.pt")]):
                    train_search.main()
            run = json.loads((output / "run.json").read_text())
            self.assertIsNone(run["config"]["checkpoint_sha256"])
            self.assertEqual(run["config"]["path_mode"], "straight_through")
            self.assertEqual(run["initial_architecture"]["per_layer"][0]["unique_preferred_counts"]["rsa"], 12)
            saved = torch.load(output / "fixed-student.pt", weights_only=True)
            self.assertEqual(saved["student_init"], "scratch")


if __name__ == "__main__":
    unittest.main()
