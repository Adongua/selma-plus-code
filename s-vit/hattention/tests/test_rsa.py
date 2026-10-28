"""RSA arithmetic tests; plaintext stand-in does not validate MPC security."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hattention import rsa
from hattention.reference import rsa_attention


class PlainRuntime:
    device = torch.device("cpu")

    def public(self, value):
        return value

    def mul(self, left, right):
        return left * right

    def exchange(self, values):
        return [torch.zeros_like(value) for value in values]

    def refresh(self, value):
        return value


class RSATests(unittest.TestCase):
    def test_profile_range_and_quantized_epsilon(self):
        profile = rsa.RSAProfile()
        self.assertEqual(profile.effective_epsilon, 1 / (1 << 17))
        with self.assertRaises(ValueError):
            rsa.RSAProfile(denominator_min=0)
        with self.assertRaises(ValueError):
            rsa.RSAProfile(denominator_max=65)
        with self.assertRaises(ValueError):
            rsa.RSAProfile(epsilon=1e-12)

    def test_plaintext_stand_in_matches_rsa_and_burns_material(self):
        base = PlainRuntime()
        wide = SimpleNamespace(base=base)
        profile = rsa.RSAProfile()
        shape = (1, 1, 17, 16)
        generator = torch.Generator().manual_seed(1)
        inputs = [0.25 * torch.randn(shape, generator=generator, dtype=torch.float64)
                  for _ in range(3)]
        shares = [torch.round(value * (1 << 17)).to(torch.int64) for value in inputs]
        quantized = [value.to(torch.float64) / (1 << 17) for value in shares]
        reference, aux = rsa_attention(*quantized, epsilon=profile.effective_epsilon,
                                       radius=profile.radius, return_aux=True)
        self.assertGreater(float(aux["denominator"].min()), profile.denominator_min)
        self.assertLess(float(aux["denominator"].max()), profile.denominator_max)

        def fake_srelu(base, wide, score, material):
            value = score.to(torch.float64) / (1 << 17)
            result = (value + torch.sqrt(value.square() + profile.radius**2)) / 2
            return torch.round(result * (1 << 17)).to(torch.int64)

        def fake_trunc(base, value, shift, material):
            return torch.div(value, 1 << shift, rounding_mode="floor")

        with patch.object(rsa, "prepare_srelu", return_value=object()), \
             patch.object(rsa, "ruler_srelu", side_effect=fake_srelu), \
             patch.object(rsa, "prob_trunc_material", return_value=None), \
             patch.object(rsa, "prob_trunc", side_effect=fake_trunc):
            material = rsa.prepare_rsa(base, wide, shape, profile)
            actual = rsa.secure_rsa(base, wide, *shares, material).double() / (1 << 17)
            self.assertLess(float((actual - reference).abs().max()), 0.002)
            with self.assertRaises(RuntimeError):
                rsa.secure_rsa(base, wide, *shares, material)


if __name__ == "__main__":
    unittest.main()
