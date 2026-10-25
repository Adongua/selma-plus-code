import unittest

import torch
from torch.nn import functional as F

from slayernorm.protocol import Profile
from slayernorm.reference import scaled_layernorm


class ReferenceTests(unittest.TestCase):
    def test_scaled_identity_and_affine(self):
        rng = torch.Generator().manual_seed(29)
        for shape in [(1, 1), (2, 17), (2, 3, 257)]:
            x = torch.randn(shape, generator=rng, dtype=torch.float64)
            gamma = torch.randn(shape[-1], generator=rng, dtype=torch.float64)
            beta = torch.randn(shape[-1], generator=rng, dtype=torch.float64)
            for scale in [.5, 1, 2, 4]:
                torch.testing.assert_close(scaled_layernorm(x, gamma, beta, scale=scale),
                    F.layer_norm(x, (shape[-1],), gamma, beta, 1e-5), rtol=1e-12, atol=1e-12)

    def test_constant_and_token_independence(self):
        x = torch.tensor([[1., 1., 1.], [-2., 0., 2.]], dtype=torch.float64)
        gamma, beta = torch.ones(3, dtype=torch.float64), torch.arange(3, dtype=torch.float64)
        actual = scaled_layernorm(x, gamma, beta)
        torch.testing.assert_close(actual[0], beta)
        torch.testing.assert_close(actual[1], scaled_layernorm(x[1], gamma, beta))

    def test_profile_guard_and_epsilon_resolution(self):
        for shape in [(1, 197), (1, 257), (1, 577), (8, 257), (2, 3, 768)]:
            Profile().validate(shape)
        self.assertEqual(Profile().epsilon_integer, 1)
        for bad in [Profile(epsilon=0), Profile(scale=0), Profile(scale=10),
                    Profile(input_bound=8), Profile(epsilon=float('nan'))]:
            with self.assertRaises(ValueError):
                bad.validate((1, 257))
        with self.assertRaises(ValueError):
            Profile().validate((0, 257))


if __name__ == "__main__":
    unittest.main()
