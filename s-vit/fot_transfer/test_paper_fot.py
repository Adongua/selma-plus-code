import unittest
import torch
from .paper_fot import Decoder, pack, unpack, random_projection
from torch.func import functional_call


class PaperFoTTest(unittest.TestCase):
    def test_decoder_functional_parameters_and_shape(self):
        torch.manual_seed(7)
        decoder = Decoder().double().eval()
        features = torch.randn(2, 768, dtype=torch.float64)
        weights = pack(decoder)
        self.assertEqual(weights.numel(), 9920)
        expected = decoder(features)
        actual = functional_call(decoder, unpack(decoder, weights), (features,))
        self.assertEqual(tuple(actual.shape), (2, 3, 224, 224))
        torch.testing.assert_close(actual, expected)

    def test_projection_is_orthogonal_to_existing_basis(self):
        g = torch.Generator().manual_seed(13)
        old = random_projection(40, 3, torch.empty(40, 0, dtype=torch.float64), g, 'cpu')
        new = random_projection(40, 4, old, g, 'cpu')
        torch.testing.assert_close(old.T @ new, torch.zeros(3, 4, dtype=torch.float64),
                                   atol=1e-12, rtol=0)
        torch.testing.assert_close(new.T @ new, torch.eye(4, dtype=torch.float64),
                                   atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
