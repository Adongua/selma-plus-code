import unittest

import torch
from torch.nn import functional as F

from sgelu.baseline import puma_gelu_reference


class PumaReferenceTests(unittest.TestCase):
    def test_piecewise_boundaries_and_extremes(self):
        x = torch.tensor([-8.0, -4.0, -1.0, 0.0, 1.0, 3.0, 8.0])
        y = puma_gelu_reference(x)
        self.assertEqual(y[0].item(), 0.0)
        self.assertEqual(y[-1].item(), 8.0)
        self.assertTrue(torch.isfinite(y).all())

    def test_published_approximation_error_on_supported_range(self):
        x = torch.linspace(-8.0, 8.0, 10001)
        error = (puma_gelu_reference(x) - F.gelu(x, approximate="none")).abs().max()
        self.assertLess(float(error), 0.021)
