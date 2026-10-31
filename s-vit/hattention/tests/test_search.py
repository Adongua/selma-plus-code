"""Correctness checks for plaintext structure search and cost accounting."""

import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hattention.config import HAttentionConfig
from hattention.calibrate_group_costs import all_pairs, parse_pair, validate_report
from hattention.group_costs import assemble_reports, load_complete_tables, METADATA
from hattention.search import (alternating_step, architecture_parameters, discretize,
                               expected_group_cost, export_fixed_student,
                               grouped_count_distribution, make_search_student,
                               search_layers, temperature_at, weight_parameters)
from hattention.student_checkpoint import configure_fixed_rsa_search


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)
        self.attn_drop = nn.Identity()
        self.proj_drop = nn.Identity()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, x):
        return self.self_attn(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.seq_pool = False
        self.classifier.blocks = nn.ModuleList([Block()])

    def forward(self, x):
        for block in self.classifier.blocks:
            x = block(x)
        return x


def table():
    return torch.tensor([[1., 0.5, 0.], [3., 2., float("nan")],
                         [5., float("nan"), float("nan")]])


class SearchTests(unittest.TestCase):
    def test_fixed_rsa_search_masks_gradients_warmup_and_export(self):
        torch.manual_seed(7)
        source = Model()
        source.classifier.blocks[0].self_attn.alpha = nn.Parameter(torch.tensor([[[[1.]], [[0.]]]]))
        model = make_search_student(source, [HAttentionConfig(2, ("rsa", "lsa"), patch_grid=(2, 2))])
        paths = configure_fixed_rsa_search(model, source, mixer_probability=.05)
        layer = search_layers(model)[0]
        self.assertEqual(paths, [["rsa", "lsa"]])
        self.assertTrue(torch.allclose(layer.probabilities(), torch.tensor([[1., 0., 0.], [0., .95, .05]])))
        weights = torch.optim.AdamW(weight_parameters(model), lr=1e-3)
        architecture = torch.optim.Adam(architecture_parameters(model), lr=1e-2)
        value = torch.randn(2, 5, 8) * .2
        closure = lambda: model(value).square().mean()
        before = layer.structure_logits.detach().clone()
        alternating_step(model, weights, architecture, closure, closure,
                         {"time": [table()]}, {"time": 10.}, update_architecture=False)
        self.assertTrue(torch.equal(before, layer.structure_logits))
        alternating_step(model, weights, architecture, closure, closure,
                         {"time": [table()]}, {"time": 10.})
        self.assertFalse(torch.equal(before[1], layer.structure_logits[1]))
        self.assertTrue(torch.equal(before[0], layer.structure_logits[0]))
        # A low observed RSA denominator must be reported, not cause the fixed
        # RSA head to be silently replaced by an unrelated path.
        layer.observed_min_denominator.fill_(0)
        result = discretize([layer], {"time": [table()]}, {"time": 2.})
        self.assertEqual(result["head_types"], [["rsa", "mixer"]])
        fixed = export_fixed_student(model, result["head_types"])
        self.assertTrue(torch.isfinite(fixed(value)).all())
        self.assertNotIn("classifier.blocks.0.self_attn.allowed_paths", model.state_dict())

    def test_group_distribution_matches_enumeration_and_has_gradients(self):
        logits = torch.tensor([[0.2, -0.3, 0.1], [-0.4, 0.3, 0.7]], requires_grad=True)
        probs = logits.softmax(-1)
        distribution = grouped_count_distribution(probs)
        reference = torch.zeros_like(distribution)
        for paths in itertools.product(range(3), repeat=2):
            reference[paths.count(0), paths.count(2)] += probs[0, paths[0]] * probs[1, paths[1]]
        self.assertTrue(torch.allclose(distribution, reference))
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=6)
        cost = expected_group_cost(probs, table())
        cost.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_search_update_budget_repair_and_fixed_export(self):
        torch.manual_seed(11)
        config = HAttentionConfig(2, ("rsa", "lsa"), patch_grid=(2, 2))
        source = Model()
        model = make_search_student(source, [config])
        layer = search_layers(model)[0]
        value = torch.randn(2, 5, 8) * 0.2
        weight_opt = torch.optim.AdamW(weight_parameters(model), lr=1e-3)
        arch_opt = torch.optim.Adam(architecture_parameters(model), lr=1e-2)
        before = layer.structure_logits.detach().clone()
        closure = lambda: model(value).square().mean()
        result = alternating_step(model, weight_opt, arch_opt, closure, closure,
                                  {"online_seconds": [table()]}, {"online_seconds": 1.0})
        self.assertTrue(torch.isfinite(torch.tensor(result["architecture_loss"])))
        self.assertFalse(torch.equal(before, layer.structure_logits))
        with torch.no_grad():
            layer.structure_logits[:] = torch.tensor([[4., 0., -1.], [4., 0., -1.]])
        selection = discretize([layer], {"online_seconds": [table()]}, {"online_seconds": 1.0})
        self.assertLessEqual(selection["costs"]["online_seconds"], 1.0)
        fixed = export_fixed_student(model, selection["head_types"])
        self.assertEqual(len(search_layers(fixed)), 0)
        self.assertEqual(fixed.classifier.blocks[0].self_attn.config.head_types,
                         tuple(selection["head_types"][0]))
        self.assertTrue(torch.isfinite(fixed(value)).all())
        self.assertTrue(torch.equal(model.classifier.blocks[0].self_attn.qkv.weight,
                                    fixed.classifier.blocks[0].self_attn.qkv.weight))
        self.assertEqual(temperature_at(0, 2), 5.0)
        self.assertAlmostEqual(temperature_at(1, 2), 0.1)

    def test_unobserved_rsa_is_not_selected_and_missing_cost_rejected(self):
        config = HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))
        layer = search_layers(make_search_student(Model(), [config]))[0]
        with torch.no_grad():
            layer.structure_logits[:, 0] = 10
        result = discretize([layer], {"time": [table()]}, {"time": 10.0})
        self.assertNotIn("rsa", result["head_types"][0])
        invalid = table().clone()
        invalid[0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            discretize([layer], {"time": [invalid]}, {"time": 10.0})

    def test_one_hot_search_matches_exported_fixed_paths_with_custom_scale(self):
        torch.manual_seed(29)
        config = HAttentionConfig(2, ("rsa", "mixer"), rho=0.9, patch_grid=(2, 2))
        model = make_search_student(Model(), [config], temperature=1).eval()
        layer = search_layers(model)[0]
        with torch.no_grad():
            layer.structure_logits[:] = torch.tensor([[30., -30., -30.], [-30., -30., 30.]])
        value = torch.randn(2, 5, 8)
        expected = model(value)
        fixed = export_fixed_student(model, [["rsa", "mixer"]]).eval()
        self.assertTrue(torch.allclose(expected, fixed(value), atol=1e-6, rtol=1e-6))

    def test_impossible_budget_reports_repair_failure(self):
        config = HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))
        layer = search_layers(make_search_student(Model(), [config]))[0]
        with self.assertRaisesRegex(RuntimeError, "no structure meets all budgets"):
            discretize([layer], {"time": [table() + 1]}, {"time": 0.1})

    def test_global_discretization_escapes_non_monotone_single_head_trap(self):
        config = HAttentionConfig(2, ("rsa", "rsa"), patch_grid=(2, 2))
        layer = search_layers(make_search_student(Model(), [config]))[0]
        with torch.no_grad():
            layer.observed_min_denominator.fill_(1.0)
            layer.structure_logits[:, 0] = 8.0
            layer.structure_logits[:, 1:] = -8.0
        non_monotone = torch.tensor([
            [0.0, 20.0, 20.0],
            [6.0, 100.0, float("nan")],
            [5.0, float("nan"), float("nan")],
        ])
        selected = discretize([layer], {"time": [non_monotone]}, {"time": 1.0})
        self.assertEqual(selected["head_types"], [["lsa", "lsa"]])
        self.assertEqual(selected["costs"]["time"], 0.0)

    def test_cls_constraint_preserves_connected_pareto_state(self):
        config = HAttentionConfig(2, ("mixer", "mixer"), patch_grid=(2, 2))
        layer = search_layers(make_search_student(Model(), [config]))[0]
        with torch.no_grad():
            layer.structure_logits[:] = torch.tensor([[-10., -5., 10.], [-10., -5., 10.]])
        selected = discretize([layer, layer], {"time": [table(), table()]}, {"time": 0.5})
        self.assertEqual(sum(path != "mixer" for paths in selected["head_types"] for path in paths), 1)
        self.assertEqual(selected["costs"]["time"], 0.5)
        with self.assertRaisesRegex(RuntimeError, "connects CLS"):
            discretize([layer], {"time": [table()]}, {"time": 0.1})
        with self.assertRaisesRegex(ValueError, "all-Mixer"):
            export_fixed_student(make_search_student(Model(), [config]), [["mixer", "mixer"]])


class GroupCostTests(unittest.TestCase):
    def test_calibration_configuration_checks_shape_grid_and_order(self):
        self.assertEqual(len(all_pairs(12)), 91)
        self.assertEqual(parse_pair("2,1"), (2, 1))
        report = {"backend": "gloo", "world_size": 3, "shape_b_h_n_d": [1, 2, 17, 8],
                  "repeats": 1, "seed": 42, "input_precision_bits": 17,
                  "ring3_precision_bits": 17, "has_cls_token": True,
                  "patch_grid": [4, 4], "head_types": {"hybrid": ["lsa", "mixer"]},
                  "methods": ["hybrid"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            args = SimpleNamespace(backend="gloo", repeats=1, seed=42)
            validate_report(path, (1, 2, 17, 8), ["lsa", "mixer"], args)
            with self.assertRaisesRegex(ValueError, "head order"):
                validate_report(path, (1, 2, 17, 8), ["mixer", "lsa"], args)
            report["patch_grid"] = None
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "patch_grid"):
                validate_report(path, (1, 2, 17, 8), ["lsa", "mixer"], args)

    def test_report_assembly_requires_full_coverage_and_rejects_failures(self):
        metadata = {key: "fixture" for key in METADATA}
        metadata.update(backend="gloo", world_size=3, shape_b_h_n_d=[1, 1, 5, 4],
                        patch_grid=[2, 2], has_cls_token=True, rho=0.5, rsa_profile={},
                        input_precision_bits=17, ring3_precision_bits=17, input_std=0.25,
                        seed=1, repeats=1, max_abs_error_threshold=0.005)
        phase = {"seconds_max": 0.1, "system_bytes": 16, "rounds_per_rank": [1, 1, 1]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for method in ("rsa", "lsa", "mixer"):
                path = root / f"{method}.json"
                report = {**metadata, "status": "benchmark_passed_not_security_validated",
                          "head_types": {"hybrid": [method]}, "records": [
                              {"warmup": False, "methods": {"hybrid": {
                                  "correctness_passed": True, "online": phase, "offline": phase}}}]}
                path.write_text(json.dumps(report))
                paths.append(path)
            partial = assemble_reports(paths[:1])
            self.assertFalse(partial["complete"])
            complete = assemble_reports(paths)
            self.assertTrue(complete["complete"])
            output = root / "table.json"
            output.write_text(json.dumps(complete))
            tables, _ = load_complete_tables(output)
            self.assertAlmostEqual(float(tables["online_seconds"][1, 0]), 0.1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                assemble_reports([paths[0], paths[0]])


if __name__ == "__main__":
    unittest.main()
