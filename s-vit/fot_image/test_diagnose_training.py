import unittest
import torch
from .diagnose_training import directional_probe


class GradientDiagnosticTests(unittest.TestCase):
    def test_quadratic_directional_derivative_is_exact(self):
        x=torch.linspace(-.2,.2,91,dtype=torch.float64)
        target=torch.linspace(.5,-.1,91,dtype=torch.float64)
        def loss(value): return ((value-target)**2).sum()/2
        report=directional_probe(loss,x,[.01],directions=4,repeats=2,seed=55)
        self.assertGreater(report['true_gradient_norm'],0)
        self.assertLess(report['measurements'][0]['directional_relative_rmse'],1e-10)
        self.assertIn('4',report['measurements'][0]['mean_cosine_by_directions'])

    def test_zero_gradient_is_not_misreported(self):
        with self.assertRaisesRegex(ArithmeticError,'near-zero'):
            directional_probe(lambda z:(z*z).sum(),torch.zeros(91,dtype=torch.float64),[.01],1,1,1)


if __name__=='__main__': unittest.main()
