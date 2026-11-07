import unittest
import torch
from training.core import ZOConfig, PlainAdam, basis, direction, weights
from training.trusted import split_q24, unpack


class CoreTests(unittest.TestCase):
    def test_basis_and_direction(self):
        p = ZOConfig(rank=4)
        b = basis(64, p)
        torch.testing.assert_close(b.t()@b, torch.eye(4, dtype=torch.float64))
        torch.testing.assert_close(direction(1, p).norm(), torch.tensor(1., dtype=torch.float64))
        torch.testing.assert_close(direction(1, p), direction(1, p))
        # Fixed-point public basis keeps approximate orthonormality.
        quant = torch.round(b*2**24)/2**24
        self.assertLess((quant.t()@quant-torch.eye(4)).abs().max(), 1e-5)

    def test_matches_torch_adam(self):
        p = ZOConfig()
        adam = PlainAdam(p)
        z = torch.zeros(p.rank, dtype=torch.float64, requires_grad=True)
        reference = torch.optim.Adam([z], lr=p.lr, betas=(p.beta1,p.beta2), eps=p.epsilon)
        for step, delta in enumerate([.01, -.005, .002, .0], 1):
            u = direction(step, p)
            z.grad = p.rank*delta/(2*p.sigma)*u
            reference.step()
            actual = adam.update(delta, u)
            torch.testing.assert_close(actual, z)

    def test_resharing(self):
        x = torch.tensor([.1, -.5, 0., 3.], dtype=torch.float64)
        shares = [unpack(v, x.shape) for v in split_q24(x)]
        torch.testing.assert_close(sum(shares).double()/2**24, x, atol=1e-7, rtol=0)

    def test_contracts(self):
        with self.assertRaises(ValueError):
            ZOConfig(sigma=0).validate()
        with self.assertRaises(ValueError):
            weights([0,0,0])
        with self.assertRaises(ArithmeticError):
            PlainAdam(ZOConfig()).update(100, direction(1, ZOConfig()))


if __name__ == '__main__':
    unittest.main()
