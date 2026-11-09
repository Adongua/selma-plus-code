import unittest
import torch
from .plaintext_ops import AccuracyPlainOps


class AccuracyPlainOpsTests(unittest.TestCase):
    def test_relaxes_only_finite_gelu_and_records_occurrences(self):
        ops = AccuracyPlainOps({}, allow_gelu_out_of_range=True)
        ops.check(torch.tensor([-13., 2., 14.]), 12, 'b0.gelu_input')
        report = ops.range_report()
        self.assertEqual(report['gelu_out_of_range_calls'], 1)
        self.assertEqual(report['gelu_out_of_range_elements'], 2)
        self.assertEqual(report['gelu_max_abs'], 14)
        with self.assertRaises(ArithmeticError):
            ops.check(torch.tensor([float('nan')]), 12, 'b0.gelu_input')
        with self.assertRaises(ArithmeticError):
            ops.check(torch.tensor([20.]), 16, 'b0.q')

    def test_default_is_strict(self):
        ops = AccuracyPlainOps({})
        with self.assertRaises(ArithmeticError):
            ops.check(torch.tensor([13.]), 12, 'b0.gelu_input')


if __name__ == '__main__': unittest.main()
