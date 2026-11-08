import unittest
import torch

from .conditioning import LocalEncoder, conditioned_latent, direction


class ConditioningTests(unittest.TestCase):
    def test_conditioning_depends_on_feature_and_has_zero_initial_prompt(self):
        rank, features = 4, 3
        weights = torch.zeros(rank * (features + 1), dtype=torch.float64)
        a = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
        self.assertTrue(torch.equal(conditioned_latent(weights, a, rank, features), torch.zeros(rank, dtype=torch.float64)))
        weights[rank] = 2
        self.assertNotEqual(conditioned_latent(weights, a, rank, features)[0], conditioned_latent(weights, b, rank, features)[0])

    def test_encoder_and_direction_reproducible(self):
        encoder = LocalEncoder('fixture', 8, 'cpu')
        x = torch.randn(1, 3, 4, 4, dtype=torch.float64)
        torch.testing.assert_close(encoder(x), encoder(x))
        u = direction(4, 36, 7)
        torch.testing.assert_close(u, direction(4, 36, 7))
        self.assertAlmostEqual(u.norm().item(), 1.)


if __name__ == '__main__': unittest.main()
