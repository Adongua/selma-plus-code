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
from hattention import ranked_mixer
from hattention.student_checkpoint import make_checkpoint_search_student
from hattention.reference import rsa_attention
from hattention.search import discretize, search_objective
from hattention.search import export_fixed_student
from hattention.train_rsa_warmup import balanced_subset, train_epoch as warmup_epoch
from hattention.train_search import load_rsa_warmup_weights


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
    def test_ranked_mixer_entrypoint_exports_and_recovers_independent_candidates(self):
        items = [("a", 0), ("b", 1)]
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        teacher_meta = {"teacher_hidden_size": 6, "weights_sha256": "w", "config_sha256": "c",
                        "processor_sha256": "p"}
        def load_source(*args):
            source = WideStudent()
            source.classifier.blocks[0].self_attn.alpha = torch.tensor([1.] * 6 + [0.] * 6)
            return source, {"sha256": "student"}
        def make_model(source, **kwargs):
            return make_search_student(source, [HAttentionConfig(12, ("lsa",) * 12, patch_grid=(2, 2))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = load_source()
            model = make_model(source)
            ranked_mixer.configure_fixed_rsa_search(model, source, direct_alpha=True)
            model.search_feature_projection = nn.Linear(192, 6)
            config = {"optimization": "joint_alpha_lasso_v1", "search_space": "fixed_rsa_lsa_mixer",
                      "checkpoint_sha256": "student", "cost_table_sha256": "cost", "seed": 1,
                      "training_split": {"seed": 1}, "validation_per_class": 1,
                      "search_holdout": {"holdout": 1}, "mapping": "first-name",
                      "mixer_init_probability": .05, "teacher_weights_sha256": "w",
                      "teacher_config_sha256": "c", "teacher_processor_sha256": "p"}
            checkpoint = root / "search.pt"
            torch.save({"format": "hattention_search_checkpoint_v1", "config": config,
                        "model": model.state_dict()}, checkpoint)
            output = root / "output"
            argv = ["ranked_mixer", "--search-checkpoint", str(checkpoint), "--checkpoint", str(root / "model"),
                    "--mpcvit-root", str(root), "--dataset-root", str(root), "--teacher-dir", str(root),
                    "--cost-table", str(root / "cost"), "--output-root", str(output), "--mixer-counts", "0", "1",
                    "--recovery-steps", "1", "--eval-samples", "2", "--device", "cpu"]
            table = torch.zeros(13, 13, dtype=torch.float64)
            def digest(path):
                if Path(path).name == "model": return "student"
                if Path(path).name == "cost": return "cost"
                return "search"
            with patch("sys.argv", argv), patch.object(ranked_mixer, "file_sha256", side_effect=digest), \
                 patch.object(ranked_mixer, "training_splits", return_value=(["a", "b"], items, items, {"seed": 1})), \
                 patch.object(ranked_mixer, "split_architecture_items", return_value=(items, items, {"holdout": 1})), \
                 patch.object(ranked_mixer, "load_student", side_effect=load_source), \
                 patch.object(ranked_mixer, "make_checkpoint_search_student", side_effect=make_model), \
                 patch.object(ranked_mixer, "load_teacher", return_value=(teacher, teacher_meta)), \
                 patch.object(ranked_mixer, "load_complete_tables", return_value=(
                     {"online_seconds": table}, {"shape_b_h_n_d": [1, 12, 257, 16], "world_size": 3})), \
                 patch.object(ranked_mixer, "SearchDataset", WideFixture), redirect_stdout(io.StringIO()):
                ranked_mixer.main()
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "completed_plaintext_not_security_validated")
            self.assertEqual([row["mixer_count"] for row in report["candidates"]], [0, 1])
            self.assertEqual(report["candidates"][1]["path_counts"]["rsa"], 6)
            self.assertEqual(report["candidates"][1]["recovery"]["steps"], 1)
            first = torch.load(output / "mixer-000-before.pt", weights_only=True)
            second = torch.load(output / "mixer-001-before.pt", weights_only=True)
            key = "classifier.blocks.0.self_attn.qkv.weight"
            self.assertTrue(torch.equal(first["state_dict"][key], second["state_dict"][key]))
            self.assertTrue((output / "mixer-001-after.pt").exists())

    def test_fixed_rsa_entrypoint_training_resume_and_export(self):
        items = [(f"{c}/{i}", c) for c in range(2) for i in range(2)]
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        meta = {"teacher_hidden_size": 6, "weights_sha256": "w", "config_sha256": "c",
                "processor_sha256": "p"}
        table = torch.zeros(13, 13, dtype=torch.float64)
        def load_source(*args):
            source = WideStudent()
            source.classifier.blocks[0].self_attn.alpha = torch.tensor([1.] * 6 + [0.] * 6)
            return source, {"sha256": "student"}
        def make_model(source, **kwargs):
            return make_search_student(source, [HAttentionConfig(12, ("lsa",) * 12, patch_grid=(2, 2))])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cost = root / "cost.json"
            cost.write_text("{}")
            output = root / "output"
            argv = ["train_search", "--checkpoint", str(root / "model"), "--mpcvit-root", str(root),
                    "--dataset-root", str(root), "--teacher-dir", str(root), "--cost-table", str(cost),
                    "--output-root", str(output), "--epochs", "2", "--budget-online-seconds", "9",
                    "--search-space", "fixed_rsa_lsa_mixer", "--architecture-warmup-epochs", "1",
                    "--validation-per-class", "1", "--eval-samples", "2", "--max-steps-per-epoch", "1",
                    "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch.object(train_search, "training_splits", return_value=(
                     ["class0", "class1"], [("w0", 0), ("w1", 1)], items, {"seed": 1})), \
                 patch.object(train_search, "load_student", side_effect=load_source), \
                 patch.object(train_search, "make_checkpoint_search_student", side_effect=make_model), \
                 patch.object(train_search, "load_teacher", return_value=(teacher, meta)), \
                 patch.object(train_search, "load_complete_tables", return_value=(
                     {"online_seconds": table}, {"shape_b_h_n_d": [1, 12, 257, 16]})), \
                 patch.object(train_search, "SearchDataset", WideFixture), redirect_stdout(io.StringIO()):
                with patch.object(train_search, "discretize", side_effect=RuntimeError("interrupted export")):
                    with self.assertRaisesRegex(RuntimeError, "interrupted export"):
                        train_search.main()
                with patch("sys.argv", argv + ["--resume", str(output / "epoch-0002.pt")]):
                    train_search.main()
            first = json.loads((output / "epoch-0001.json").read_text())
            second = json.loads((output / "epoch-0002.json").read_text())
            self.assertFalse(first["architecture_updates_enabled"])
            self.assertTrue(second["architecture_updates_enabled"])
            self.assertTrue((output / "baseline.json").exists())
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(selection["head_types"][0][:6], ["rsa"] * 6)
            self.assertTrue(all(p in ("lsa", "mixer") for p in selection["head_types"][0][6:]))
            self.assertEqual(len(selection["fixed_rsa_observations"]), 1)
            self.assertEqual(selection["path_counts"]["rsa"], 6)
            self.assertEqual(selection["alpha_lsa_by_layer"][0][:6], [None] * 6)
            checkpoint1 = torch.load(output / "epoch-0001.pt", weights_only=True)
            checkpoint2 = torch.load(output / "epoch-0002.pt", weights_only=True)
            key = "classifier.blocks.0.self_attn.lsa_alpha"
            self.assertTrue(torch.allclose(checkpoint1["model"][key], torch.full((12,), .95)))
            self.assertFalse(torch.equal(checkpoint1["model"][key][6:], checkpoint2["model"][key][6:]))

    def test_rsa_warmup_learns_and_rejects_overfit_checkpoint_for_search(self):
        torch.manual_seed(3)
        items = [(f"{c}/{i}", c) for c in range(3) for i in range(4)]
        self.assertEqual(balanced_subset(items, 2, 2, 5), balanced_subset(items, 2, 2, 5))
        search = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))])
        fixed = export_fixed_student(search, [["rsa", "rsa"]])
        optimizer = torch.optim.AdamW(fixed.parameters(), lr=.01)
        losses = [warmup_epoch(fixed, None, Fixture(), optimizer, torch.device("cpu"),
                               batch_size=4, seed=1)["loss"]["ce"] for _ in range(15)]
        self.assertLess(losses[-1], losses[0])
        search.search_feature_projection = nn.Linear(8, 6)
        split = {"weight_manifest_sha256": "fixture"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warmup.pt"
            artifact = {"format": "hattention_rsa_warmup_v1", "config": {
                "overfit_diagnostic": True, "class_order": ["a"], "training_split": split},
                "model": fixed.state_dict(), "metrics": {"epoch": 1}}
            torch.save(artifact, path)
            with self.assertRaisesRegex(ValueError, "overfit diagnostic"):
                load_rsa_warmup_weights(search, path, ["a"], split)
            artifact["config"]["overfit_diagnostic"] = False
            torch.save(artifact, path)
            load_rsa_warmup_weights(search, path, ["a"], split)
            self.assertTrue(torch.equal(search.classifier.blocks[0].self_attn.qkv.weight,
                                        fixed.classifier.blocks[0].self_attn.qkv.weight))
            with self.assertRaisesRegex(ValueError, "dataset split"):
                load_rsa_warmup_weights(search, path, ["other"], split)

    def test_rsa_warmup_distillation_logs_components(self):
        search = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))])
        fixed = export_fixed_student(search, [["rsa", "rsa"]])
        fixed.search_feature_projection = nn.Linear(8, 6)
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        optimizer = torch.optim.AdamW(fixed.parameters(), lr=.001)
        metrics = warmup_epoch(fixed, teacher, Fixture(), optimizer, torch.device("cpu"),
                               batch_size=2, seed=2, max_steps=1)
        self.assertEqual(set(metrics["loss"]), {"total", "ce", "kd", "feature"})
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))

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
        efficient = discretize([layer], {"time": [table]}, {"time": 10.}, cost_preference=20.,
                               require_cls_mixing=False)
        self.assertEqual(likely["head_types"], [["lsa", "lsa"]])
        self.assertEqual(efficient["head_types"], [["mixer", "mixer"]])

    def test_budget_constrained_objective_does_not_reward_extra_slack(self):
        model = make_search_student(Student(), [HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))])
        layer = search_layers(model)[0]
        layer.observed_min_denominator.fill_(1)
        model(torch.randn(1, 5, 8) * .1)
        table = torch.tensor([[2., 1., 0.], [4., 3., float("nan")], [6., float("nan"), float("nan")]])
        task = torch.tensor(40.)
        under_loss, under = search_objective(
            [layer], task, {"time": [table]}, {"time": 10.}, task_loss_scale=50.,
            lambda_task=1., lambda_cost=10., lambda_efficiency=0.,
            lambda_num=0., lambda_count=0., lambda_entropy=0.)
        self.assertAlmostEqual(float(under["task_normalized"]), .8)
        self.assertEqual(float(under["budget"]), 0.)
        self.assertAlmostEqual(float(under_loss), .8)
        over_loss, over = search_objective(
            [layer], task, {"time": [table]}, {"time": 1.}, task_loss_scale=50.,
            lambda_task=1., lambda_cost=10., lambda_efficiency=0.,
            lambda_num=0., lambda_count=0., lambda_entropy=0.)
        self.assertGreater(float(over["budget"]), 0.)
        self.assertGreater(float(over_loss), float(under_loss))

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
