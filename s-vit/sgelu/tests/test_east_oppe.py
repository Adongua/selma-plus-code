import unittest

import torch
from torch.nn import functional as F

from sgelu.baseline import (
    EAST_OPPE_DEGREE,
    EAST_OPPE_SEGMENTS,
    east_oppe_3pc_gelu_reference,
    east_oppe_table,
)


class EastOppeReferenceTests(unittest.TestCase):
    def test_table_has_published_operator_dimensions(self):
        table = east_oppe_table()
        self.assertEqual(table["coefficients"].shape,
                         (EAST_OPPE_SEGMENTS, EAST_OPPE_DEGREE + 1))
        self.assertEqual(table["thresholds"].numel(), EAST_OPPE_SEGMENTS + 1)

    def test_tail_and_symmetry(self):
        x = torch.linspace(-8.0, 8.0, 4097, dtype=torch.float64)
        result = east_oppe_3pc_gelu_reference(x)
        self.assertEqual(result[0].item(), 0.0)
        self.assertEqual(result[-1].item(), 8.0)
        torch.testing.assert_close(
            result - east_oppe_3pc_gelu_reference(-x), x, atol=5e-4, rtol=0
        )

    def test_cubic_interpolation_error_on_supported_range(self):
        x = torch.linspace(-8.0, 8.0, 10001, dtype=torch.float64)
        error = (east_oppe_3pc_gelu_reference(x) - F.gelu(x, approximate="none")).abs().max()
        self.assertLess(float(error), 0.0005)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
