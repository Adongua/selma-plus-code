import unittest

import torch

from srelu import (
    METHOD_PROVENANCE,
    SReLUProfile,
    effective_signed_bits,
    method_reference,
    quantized_drelu_reference,
)


class SReLUReferenceTests(unittest.TestCase):
    def test_two_stage_formula_and_public_constants(self):
        profile = SReLUProfile(radius=0.0055)
        value = torch.linspace(-15.99, 15.99, 10001, dtype=torch.float64)
        shifted = profile.shifted_reference(value)
        corrected = profile.corrected_reference(value)
        direct = 0.5 * (value + torch.sqrt(value.square() + profile.radius**2))
        self.assertTrue(torch.allclose(corrected, direct, atol=1e-15, rtol=0.0))
        self.assertTrue(torch.all(corrected >= 0))
        self.assertAlmostEqual(float(shifted[value.numel() // 2]), 0.0, places=15)
        self.assertEqual(profile.radius_squared_q16, 2)
        self.assertEqual(profile.radius_q17, 721)
        self.assertEqual(profile.half_radius_q17, 360)

    def test_profile_rejects_out_of_range_input(self):
        profile = SReLUProfile(input_bound=16.0)
        with self.assertRaises(ValueError):
            profile.encode(torch.tensor([16.0], dtype=torch.float64))


class BaselineReferenceTests(unittest.TestCase):
    def test_effective_signed_widths(self):
        self.assertEqual(effective_signed_bits(16.0, 17), 22)
        self.assertEqual(effective_signed_bits(16.0, 8), 13)
        self.assertEqual(effective_signed_bits(16.0, 2), 7)
        with self.assertRaises(ValueError):
            effective_signed_bits(0.0, 17)

    def test_quantized_sign_probe_keeps_full_precision_payload(self):
        value = torch.tensor(
            [-0.40, -0.10, -0.01, 0.0, 0.01, 0.10, 0.40],
            dtype=torch.float64,
        )
        actual = quantized_drelu_reference(value, fraction_bits=2)
        expected = torch.tensor(
            [0.0, -0.10, -0.01, 0.0, 0.01, 0.10, 0.40],
            dtype=torch.float64,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_method_references_and_provenance(self):
        value = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        exact = method_reference(
            "hummingbird_eco",
            value,
            hummingbird_fraction_bits=8,
            bicoptor_fraction_bits=2,
        )
        self.assertTrue(torch.equal(exact, torch.relu(value)))
        self.assertIn("not the official", METHOD_PROVENANCE["bicoptor2_keybits"])
        with self.assertRaises(ValueError):
            method_reference(
                "unknown",
                value,
                hummingbird_fraction_bits=8,
                bicoptor_fraction_bits=2,
            )


if __name__ == "__main__":
    unittest.main()
