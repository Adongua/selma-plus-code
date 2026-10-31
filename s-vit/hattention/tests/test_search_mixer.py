"""Coverage of Mixer search constraints and a complete small plaintext run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from hattention import search_mixer


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv, self.proj = nn.Linear(4, 12), nn.Linear(4, 4)
        self.attn_drop, self.proj_drop = nn.Identity(), nn.Identity()
        self.alpha = nn.Parameter(torch.tensor([[[[1.0]], [[0.0]]]]))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, x):
        return x + self.self_attn(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        self.classifier.blocks = nn.ModuleList([Block(), Block()])
        self.fc = nn.Linear(4, 200)

    def forward(self, x):
        for block in self.classifier.blocks:
            x = block(x)
        return self.fc(x[:, 0]), [x]


class Fixture(Dataset):
    def __init__(self, _root, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        value = torch.randn(257, 4, generator=torch.Generator().manual_seed(index))
        return value, torch.zeros(1), self.items[index][1]


def cost_table():
    table = torch.full((3, 3), float("nan"), dtype=torch.float64)
    for rsa in range(3):
        for mixer in range(3 - rsa):
            table[rsa, mixer] = 1.0 - 0.1 * mixer
    return table


class MixerSearchTests(unittest.TestCase):
    def test_measured_cost_and_connectivity_constraint(self):
        table = cost_table()
        self.assertAlmostEqual(search_mixer.measured_cost([["rsa", "lsa"], ["mixer", "lsa"]], table), 1.9)
        rows = search_mixer.candidates([["mixer", "mixer"], ["mixer", "rsa"]], table, 3.0, 1)
        self.assertEqual(rows, [])
        rows = search_mixer.candidates([["rsa", "lsa"]], table, 0.95, 1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["estimated_online_seconds"] <= 0.95 for row in rows))

    def test_balanced_samples_are_seeded_and_interleaved(self):
        items = [(f"{c}/{i}", c) for c in range(3) for i in range(4)]
        result = search_mixer.balanced_items(items, 6, 1)
        self.assertEqual([item[1] for item in result], [0, 1, 2, 0, 1, 2])
        self.assertEqual(result, search_mixer.balanced_items(items, 6, 1))

    def test_full_stage_and_artifact_restoration(self):
        torch.manual_seed(1)
        weight = [(f"w{i}", i % 2) for i in range(8)]
        architecture = [(f"a{i}", i % 2) for i in range(8)]
        holdout = [(f"h{i}", i % 2) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cost = root / "cost.json"
            cost.write_text("{}")
            output = root / "out"
            metadata = {"shape_b_h_n_d": [1, 12, 257, 16], "world_size": 3}
            argv = ["search_mixer", "--checkpoint", str(root / "source.pt"),
                    "--mpcvit-root", str(root), "--dataset-root", str(root),
                    "--cost-table", str(cost), "--output-root", str(output),
                    "--device", "cpu", "--max-top1-drop", "1", "--max-stages", "1",
                    "--top-k", "1", "--candidate-limit", "1", "--sensitivity-samples", "2",
                    "--selection-samples", "4", "--holdout-samples", "2", "--batch-size", "2",
                    "--recovery-steps", "1", "--budget-online-seconds", "1.85"]
            with patch.object(search_mixer, "training_splits", return_value=(["a", "b"], weight, architecture, {})), \
                 patch.object(search_mixer, "split_architecture_items", return_value=(architecture, holdout, {})), \
                 patch.object(search_mixer, "SearchDataset", Fixture), \
                 patch.object(search_mixer, "load_student", side_effect=lambda *_: (Student(), {"sha256": "source"})), \
                 patch.object(search_mixer, "load_complete_tables", return_value=({"online_seconds": cost_table()}, metadata)), \
                 patch("sys.argv", argv):
                search_mixer.main()
                result = json.loads((output / "report.json").read_text())
                self.assertEqual(result["status"], "completed_plaintext_not_security_validated")
                self.assertEqual(result["final"]["path_counts"].get("mixer"), 1)
                self.assertTrue(result["target_mixers_reached"])
                self.assertFalse(result["budget_reached"])
                restored, artifact = search_mixer.load_mixer_search_model(output / "final.pt", root / "source.pt", root)
                self.assertEqual(search_mixer.head_paths(restored), artifact["head_types"])
                rejected_argv = list(argv)
                rejected_output = root / "rejected"
                rejected_argv[rejected_argv.index("--output-root") + 1] = str(rejected_output)
                rejected_argv.extend(["--min-top1", "1.0"])
                metrics = {"samples": 4, "top1": 0.5, "top5": 0.5}
                with patch("sys.argv", rejected_argv), patch.object(search_mixer, "evaluate", return_value=metrics):
                    search_mixer.main()
                rejected = json.loads((rejected_output / "report.json").read_text())
                self.assertEqual(rejected["termination"], "no_shortlisted_candidate_meets_accuracy_floor")
                self.assertFalse(rejected["target_mixers_reached"])
                self.assertEqual(rejected["final"]["path_counts"].get("mixer", 0), 0)
                argv[argv.index("--max-stages") + 1] = "2"
                argv.append("--resume")
                search_mixer.main()
                result = json.loads((output / "report.json").read_text())
                self.assertEqual(len(result["stages"]), 2)
                self.assertEqual(result["final"]["path_counts"].get("mixer"), 2)
                self.assertEqual(result["final_artifact"], "final-resume-0002.pt")
                self.assertTrue(result["budget_reached"])
                argv[argv.index("--max-stages") + 1] = "3"
                argv.extend(["--max-top1-drop", "0.2"])
                with self.assertRaisesRegex(ValueError, "resume settings"):
                    search_mixer.main()


if __name__ == "__main__":
    unittest.main()
