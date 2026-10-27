import unittest

import torch

from hattention.config import HAttentionConfig, infer_patch_grid
from hattention.module import FixedHAttention
from hattention.protocol import (
    FixedPointProfile,
    prepare_grouped_lsa_mixer,
    secure_grouped_lsa_mixer,
)
from hattention.reference import (
    grouped_hattention,
    legacy_zero_origin_srelu,
    lsa_attention,
    lsa_attention_direct,
    mixer_attention,
    positive_srelu,
    rsa_attention,
)
from srelu import SReLUProfile


class ReferencePathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261027)

    def test_lsa_reassociation_matches_direct_order(self):
        q = torch.randn(2, 3, 17, 8, dtype=torch.float64)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        rho = 8 ** -0.5
        direct = lsa_attention_direct(q, k, v, rho=rho)
        reassociated = lsa_attention(q, k, v, rho=rho)
        self.assertTrue(torch.allclose(direct, reassociated, atol=1e-12, rtol=1e-12))

    def test_positive_srelu_preserves_rsa_invariant(self):
        value = torch.linspace(-32.0, 32.0, 10001, dtype=torch.float64)
        result = positive_srelu(value, radius=0.0055)
        self.assertTrue(torch.all(result >= 0))
        legacy = legacy_zero_origin_srelu(value, radius=0.01)
        self.assertAlmostEqual(float(legacy[value.numel() // 2]), 0.0, places=12)
        self.assertLess(float(legacy[0]), 0.0)
        self.assertTrue(
            torch.allclose(
                positive_srelu(value, radius=0.01),
                legacy + 0.005,
                atol=1e-15,
                rtol=0.0,
            )
        )

    def test_rsa_rows_are_normalized_up_to_epsilon(self):
        q = torch.randn(1, 2, 17, 4, dtype=torch.float64)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output, aux = rsa_attention(q, k, v, return_aux=True)
        self.assertEqual(output.shape, v.shape)
        self.assertTrue(torch.all(aux["weights"] >= 0))
        self.assertTrue(torch.all(aux["weights"].sum(dim=-1) <= 1.0))
        self.assertTrue(torch.all(aux["denominator"] > 0))

    def test_cls_aware_square_mixer(self):
        value = torch.arange(197, dtype=torch.float64).reshape(1, 1, 197, 1)
        mixed = mixer_attention(value)
        self.assertEqual(float(mixed[0, 0, 0, 0]), 0.0)
        expected = value[:, :, 1:, :].reshape(1, 1, 14, 14, 1)
        expected = expected.transpose(2, 3).reshape(1, 1, 196, 1)
        self.assertTrue(torch.equal(mixed[:, :, 1:, :], expected))
        self.assertTrue(torch.equal(mixer_attention(mixed), value))

    def test_grid_inference_and_rejection(self):
        self.assertEqual(infer_patch_grid(197), (14, 14))
        self.assertEqual(infer_patch_grid(257), (16, 16))
        self.assertEqual(infer_patch_grid(577), (24, 24))
        with self.assertRaises(ValueError):
            infer_patch_grid(18)

    def test_grouped_paths_restore_original_head_order_and_gradients(self):
        shape = (2, 6, 17, 4)
        q = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        k = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        v = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        paths = ("mixer", "rsa", "lsa", "rsa", "mixer", "lsa")
        actual = grouped_hattention(q, k, v, paths, patch_grid=(4, 4))

        expected_heads = []
        for head, path in enumerate(paths):
            head_q = q[:, head : head + 1]
            head_k = k[:, head : head + 1]
            head_v = v[:, head : head + 1]
            if path == "rsa":
                result = rsa_attention(head_q, head_k, head_v)
            elif path == "lsa":
                result = lsa_attention(head_q, head_k, head_v, rho=4 ** -0.5)
            else:
                result = mixer_attention(head_v, patch_grid=(4, 4))
            expected_heads.append(result)
        expected = torch.cat(expected_heads, dim=1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

        actual.square().mean().backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())


class FixedModuleTests(unittest.TestCase):
    def test_forward_shape_and_public_configuration(self):
        config = HAttentionConfig(
            num_heads=4,
            head_types=("rsa", "lsa", "mixer", "lsa"),
            patch_grid=(4, 4),
        )
        module = FixedHAttention(dim=32, config=config).double()
        value = torch.randn(2, 17, 32, dtype=torch.float64)
        output = module(value)
        self.assertEqual(output.shape, value.shape)
        self.assertIn("'mixer': 1", repr(module))

    def test_invalid_head_count_is_rejected(self):
        with self.assertRaises(ValueError):
            HAttentionConfig(num_heads=3, head_types=("rsa", "lsa"))


class FixedPointTests(unittest.TestCase):
    def test_srelu_profile_two_stage_reference(self):
        profile = SReLUProfile(radius=0.0055)
        value = torch.linspace(-16.0, 16.0, 10001, dtype=torch.float64)
        shifted = profile.shifted_reference(value)
        corrected = profile.corrected_reference(value)
        direct = 0.5 * torch.sqrt(value.square() + profile.radius**2) + 0.5 * value
        self.assertTrue(torch.allclose(corrected, direct, atol=1e-15, rtol=0.0))
        self.assertTrue(torch.all(corrected >= 0))
        self.assertAlmostEqual(float(shifted[value.numel() // 2]), 0.0, places=15)
        self.assertAlmostEqual(
            float(corrected[value.numel() // 2]),
            profile.radius / 2,
            places=15,
        )
        self.assertEqual(profile.radius_squared_q16, 2)
        self.assertEqual(profile.radius_q17, 721)
        self.assertEqual(profile.half_radius_q17, 360)
        with self.assertRaises(ValueError):
            profile.encode(torch.tensor([16.0], dtype=torch.float64))

    def test_q17_roundtrip_and_bounds(self):
        profile = FixedPointProfile(
            fraction_bits=17,
            coefficient_fraction_bits=30,
            input_bound=8.0,
        )
        value = torch.linspace(-8.0, 8.0, 257, dtype=torch.float64)
        restored = profile.decode(profile.encode(value))
        self.assertLessEqual(
            float((restored - value).abs().max()),
            0.5 / profile.scale,
        )
        with self.assertRaises(ValueError):
            profile.encode(torch.tensor([8.01], dtype=torch.float64))

    def test_mixer_only_secure_entry_is_a_local_share_permutation(self):
        class LocalRuntime:
            device = torch.device("cpu")

        runtime = LocalRuntime()
        value = torch.arange(2 * 17 * 3, dtype=torch.int64).reshape(1, 2, 17, 3)
        material = prepare_grouped_lsa_mixer(
            runtime,
            value.shape,
            ("mixer", "mixer"),
        )
        output = secure_grouped_lsa_mixer(
            runtime,
            torch.zeros_like(value),
            torch.zeros_like(value),
            value,
            material,
            patch_grid=(4, 4),
        )
        self.assertTrue(torch.equal(output, mixer_attention(value, patch_grid=(4, 4))))

    def test_secure_grouped_entry_rejects_rsa_before_material_generation(self):
        with self.assertRaises(NotImplementedError):
            prepare_grouped_lsa_mixer(
                object(),
                (1, 2, 17, 4),
                ("lsa", "rsa"),
            )


if __name__ == "__main__":
    unittest.main()
