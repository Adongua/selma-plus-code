"""Check quantized sign thresholds independently of PyTorch and MPC runtime."""

import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "srelu" / "quantized_sign.py"
spec = importlib.util.spec_from_file_location("srelu_quantized_sign", path)
quantized_sign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantized_sign)


class QuantizedSignTests(unittest.TestCase):
    def test_nearest_even_zero_boundary_at_all_precisions(self):
        input_bits = 17
        for bits in range(input_bits + 1):
            offset = quantized_sign.quantized_sign_offset(input_bits, bits)
            offset_integer = int(offset * (1 << input_bits))
            boundary = -offset_integer
            samples = {-(16 << input_bits) + 1, -1, 0, 1, (16 << input_bits) - 1}
            samples.update(boundary + delta for delta in range(-3, 4))
            for encoded in samples:
                expected = round(encoded / (1 << (input_bits - bits))) >= 0
                self.assertEqual(encoded + offset_integer >= 0, expected,
                                 (bits, encoded))

    def test_invalid_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            quantized_sign.quantized_sign_offset(17, 18)


if __name__ == "__main__":
    unittest.main()
