"""Fixed-structure distillation provenance, training and resume."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hattention import distill_fixed
from hattention.config import HAttentionConfig
from hattention.search import export_fixed_student, make_search_student
from hattention.student_checkpoint import configure_fixed_rsa_search
from hattention.teacher_guidance import TeacherGuidance
from test_train_search import Processor, Teacher, WideFixture, WideStudent


class DistillFixedTests(unittest.TestCase):
    def test_training_resume_and_mismatched_export(self):
        items = [("a", 0), ("b", 1)]
        teacher = TeacherGuidance(Teacher(), Processor(), list(range(200)))
        teacher_meta = {"teacher_hidden_size": 6, "weights_sha256": "w", "config_sha256": "c",
                        "processor_sha256": "p", "class_order": ["a", "b"]}

        def source(*args):
            model = WideStudent()
            model.classifier.blocks[0].self_attn.alpha = torch.tensor([1.] * 6 + [0.] * 6)
            return model, {"sha256": "student"}

        def make_search(model):
            return make_search_student(model, [HAttentionConfig(12, ("lsa",) * 12, patch_grid=(2, 2))])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(42)
            original, _ = source()
            model = make_search(original)
            configure_fixed_rsa_search(model, original, direct_alpha=True, layer_alpha=True)
            model.search_feature_projection = nn.Linear(192, 6)
            paths = [["rsa"] * 6 + ["mixer"] * 6]
            fixed = export_fixed_student(model, paths)
            config = {"search_space": "fixed_rsa_layer_lsa_mixer", "optimization": "joint_layer_alpha_lasso_v1",
                      "checkpoint_sha256": "student", "cost_table_sha256": "cost", "seed": 1,
                      "training_split": {"seed": 1}, "validation_per_class": 1,
                      "search_holdout": {"holdout": 1}, "mapping": "first-name",
                      "mixer_init_probability": .05, "teacher_weights_sha256": "w",
                      "teacher_config_sha256": "c", "teacher_processor_sha256": "p"}
            search = root / "epoch-0005.pt"
            torch.save({"format": "hattention_search_checkpoint_v1", "config": config,
                        "model": model.state_dict()}, search)
            fixed_path = root / "fixed.pt"
            torch.save({"format": "hattention_fixed_student_v1", "head_types": paths,
                        "state_dict": fixed.state_dict(), "source_checkpoint_sha256": "student"}, fixed_path)
            selection = root / "selection.json"
            selection.write_text(json.dumps({"head_types": paths, "source_checkpoint": str(search),
                                              "cost_table_sha256": "cost", "costs": {"online_seconds": 1}}))

            def load_fixed(*args):
                fresh_source, _ = source()
                search_model = make_search(fresh_source)
                configure_fixed_rsa_search(search_model, fresh_source, direct_alpha=True, layer_alpha=True)
                search_model.search_feature_projection = nn.Linear(192, 6)
                candidate = export_fixed_student(search_model, paths)
                candidate.load_state_dict(torch.load(fixed_path, weights_only=True)["state_dict"])
                return candidate, {"head_types": paths, "source": {"sha256": "student"}}

            def digest(path):
                return Path(path).name

            def run(output, epochs, extra=()):
                argv = ["distill_fixed", "--fixed-student", str(fixed_path), "--search-checkpoint", str(search),
                        "--selection", str(selection), "--checkpoint", str(root / "original"), "--mpcvit-root", str(root),
                        "--dataset-root", str(root), "--teacher-dir", str(root), "--output-root", str(output),
                        "--epochs", str(epochs), "--batch-size", "1", "--max-steps-per-epoch", "1",
                        "--eval-samples", "2", "--device", "cpu", *extra]
                with patch("sys.argv", argv), patch.object(distill_fixed, "load_student", side_effect=source), \
                     patch.object(distill_fixed, "make_checkpoint_search_student", side_effect=make_search), \
                     patch.object(distill_fixed, "training_splits", return_value=(["a", "b"], items, items, {"seed": 1})), \
                     patch.object(distill_fixed, "split_architecture_items", return_value=(items, items, {"holdout": 1})), \
                     patch.object(distill_fixed, "load_teacher", return_value=(teacher, teacher_meta)), \
                     patch.object(distill_fixed, "load_fixed_student", side_effect=load_fixed), \
                     patch.object(distill_fixed, "SearchDataset", WideFixture), \
                     patch.object(distill_fixed, "file_sha256", side_effect=digest), contextlib.redirect_stdout(io.StringIO()):
                    distill_fixed.main()

            run(root / "out", 1)
            report = json.loads((root / "out/report.json").read_text())
            self.assertEqual(report["path_counts"], {"rsa": 6, "mixer": 6})
            self.assertEqual(report["epochs"][0]["steps"], 1)
            trained = torch.load(root / "out/epoch-0001.pt", weights_only=True)
            self.assertEqual(trained["head_types"], paths)
            self.assertFalse(torch.equal(trained["state_dict"]["fc.weight"], fixed.state_dict()["fc.weight"]))
            run(root / "resumed", 2, ("--resume", str(root / "out/epoch-0001.pt")))
            resumed = torch.load(root / "resumed/epoch-0002.pt", weights_only=True)
            self.assertEqual(resumed["completed_epochs"], 2)
            run(root / "full", 2)
            uninterrupted = torch.load(root / "full/epoch-0002.pt", weights_only=True)
            for key in resumed["state_dict"]:
                self.assertTrue(torch.equal(resumed["state_dict"][key], uninterrupted["state_dict"][key]), key)
            bad = torch.load(fixed_path, weights_only=True)
            bad["state_dict"]["fc.bias"] += 1
            torch.save(bad, fixed_path)
            with self.assertRaisesRegex(ValueError, "do not match"):
                run(root / "bad", 1)


class OfficialEvaluationTests(unittest.TestCase):
    def test_balanced_subset_and_no_overwrite(self):
        from hattention import evaluate_fixed

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(192, 200)
                self.classifier = nn.Module()
                self.classifier.blocks = nn.ModuleList([])

            def forward(self, inputs):
                return self.fc(inputs[:, 0]), []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wnids.txt").write_text("\n".join(f"n{i:08d}" for i in range(200)))
            artifact = root / "fixed.pt"
            torch.save({"distillation_settings": {"class_order": [f"n{i:08d}" for i in range(200)]}}, artifact)
            paths = [["rsa", "mixer"]]
            meta = {"sha256": "fixed", "source": {"sha256": "source"}, "head_types": paths}
            items = [(f"img{i}", i % 2) for i in range(6)]
            output = root / "evaluation.json"
            argv = ["evaluate_fixed", "--fixed-student", str(artifact), "--checkpoint", str(root / "original"),
                    "--mpcvit-root", str(root), "--dataset-root", str(root), "--output", str(output),
                    "--max-samples", "4", "--device", "cpu"]
            with patch("sys.argv", argv), patch.object(evaluate_fixed, "validation_items", return_value=items), \
                 patch.object(evaluate_fixed, "load_fixed_student", return_value=(Classifier(), meta)), \
                 patch.object(evaluate_fixed, "SearchDataset", WideFixture), contextlib.redirect_stdout(io.StringIO()):
                evaluate_fixed.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["evaluation"]["samples"], 4)
            self.assertEqual(report["evaluation"]["scope"], "balanced official validation subset")
            self.assertEqual(report["path_counts"], {"rsa": 1, "mixer": 1})


if __name__ == "__main__":
    unittest.main()
