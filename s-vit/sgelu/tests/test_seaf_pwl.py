import unittest

import torch
from torch.nn import functional as F

from sgelu.baseline import (
    SEAF_PWL_INTERCEPT_FRACTION_BITS,
    SEAF_PWL_SEGMENTS,
    SEAF_PWL_SLOPE_FRACTION_BITS,
    seaf_pwl_3pc_gelu_reference,
    seaf_pwl_table,
)


class SeafPwlReferenceTests(unittest.TestCase):
    def test_table_has_published_seaf1_quantization_shape(self):
        table = seaf_pwl_table()
        self.assertEqual(table["slopes"].numel(), SEAF_PWL_SEGMENTS)
        self.assertEqual(table["intercepts"].numel(), SEAF_PWL_SEGMENTS)
        self.assertEqual(table["thresholds"].numel(), SEAF_PWL_SEGMENTS)
        torch.testing.assert_close(
            table["slopes"] * (1 << SEAF_PWL_SLOPE_FRACTION_BITS),
            (table["slopes"] * (1 << SEAF_PWL_SLOPE_FRACTION_BITS)).round(),
        )
        torch.testing.assert_close(
            table["intercepts"] * (1 << SEAF_PWL_INTERCEPT_FRACTION_BITS),
            (table["intercepts"] * (1 << SEAF_PWL_INTERCEPT_FRACTION_BITS)).round(),
        )

    def test_even_residual_preserves_gelu_symmetry(self):
        x = torch.linspace(-8.0, 8.0, 4097, dtype=torch.float64)
        result = seaf_pwl_3pc_gelu_reference(x)
        torch.testing.assert_close(result - seaf_pwl_3pc_gelu_reference(-x), x)

    def test_supported_range_is_a_reasonable_gelu_approximation(self):
        x = torch.linspace(-8.0, 8.0, 10001, dtype=torch.float64)
        error = (seaf_pwl_3pc_gelu_reference(x) - F.gelu(x, approximate="none")).abs().max()
        # This bound covers the deliberately low-precision (f_a, f_d)=(7,12)
        # coefficients rather than silently reporting an unquantized PWL fit.
        self.assertLess(float(error), 0.014)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
