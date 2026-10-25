import math
import random
import unittest

import torch
from torch.nn import functional as F

from sgelu.sgelu import MU0, SmoothGeLU, calibrate_mu, g1, g2
from sgelu.wide import (add96, sub96, mul96, to_limbs, limbs_from_int64,
                            limbs_from_uint64, isqrt_from_limbs, pack_wire, unpack_wire)


def from_limbs(x):
    return [sum(int(limb) << (24 * j) for j, limb in enumerate(row)) for row in x.tolist()]


class FloatG2Tests(unittest.TestCase):
    def test_origin_symmetry_and_taylor(self):
        x = torch.linspace(-8, 8, 2001, dtype=torch.float64)
        self.assertTrue(torch.allclose(g2(x) - g2(-x), x, atol=1e-14, rtol=0))
        zero = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        y = g2(zero)
        d1, = torch.autograd.grad(y, zero, create_graph=True)
        d2, = torch.autograd.grad(d1, zero)
        self.assertEqual(y.item(), 0.)
        self.assertEqual(d1.item(), .5)
        self.assertAlmostEqual(d2.item(), math.sqrt(2 / math.pi), places=13)

    def test_tail_order(self):
        x = torch.tensor([32., 64., 128.], dtype=torch.float64)
        for mu in [.7, MU0, 2.]:
            residual1 = x - g1(x, mu)
            residual2 = x - g2(x, mu)
            self.assertAlmostEqual((residual1[-2] / residual1[-1]).item(), 2, delta=.01)
            self.assertAlmostEqual((residual2[-2] / residual2[-1]).item(), 8, delta=.06)
            coefficient = (residual2[-1] * x[-1]**3).item()
            self.assertAlmostEqual(coefficient, 3 * mu**4 / 4, delta=.002 * mu**4)

    def test_independent_scalar_formula(self):
        mu = 1.3
        values = [-8, -2, -.01, 0, .01, 2, 8]
        expected = [4 / 3 * (x / 2 + x*x / (2*math.sqrt(x*x+mu*mu)))
                    - 1 / 3 * (x / 2 + x*x / (2*math.sqrt(x*x+4*mu*mu))) for x in values]
        torch.testing.assert_close(g2(torch.tensor(values, dtype=torch.float64), mu),
                                   torch.tensor(expected, dtype=torch.float64))

    def test_public_calibration_and_module(self):
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(4096, generator=generator, dtype=torch.float64)
        fitted = calibrate_mu(x)
        self.assertLess(fitted.fitted_mse, fitted.initial_mse)
        self.assertGreaterEqual(fitted.mu, .25)
        self.assertLessEqual(fitted.mu, 4)
        self.assertEqual(fitted.sample_count, x.numel())
        module = SmoothGeLU(fitted.mu)
        self.assertEqual(list(module.parameters()), [])
        torch.testing.assert_close(module(x), g2(x, fitted.mu))
        clone = SmoothGeLU()
        clone.load_state_dict(module.state_dict())
        torch.testing.assert_close(clone(x), module(x))
        self.assertEqual(module(x.float()).dtype, torch.float32)

    def test_validation_and_degenerate_fit(self):
        for bad in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                g2(torch.ones(1), bad)
        for bad in [torch.empty(0), torch.tensor([float("nan")])]:
            with self.assertRaises(ValueError):
                calibrate_mu(bad)
        self.assertEqual(calibrate_mu(torch.zeros(8)).mu, MU0)


class WideTests(unittest.TestCase):
    def test_ten_thousand_arithmetic_and_wire(self):
        rng = random.Random(43)
        modulus = 1 << 96
        xs = [rng.randrange(modulus) for _ in range(10000)]
        ys = [rng.randrange(modulus) for _ in range(10000)]
        x, y = torch.stack([to_limbs(n) for n in xs]), torch.stack([to_limbs(n) for n in ys])
        for operation, expected in [(add96, [(a+b) % modulus for a,b in zip(xs,ys)]),
                                    (sub96, [(a-b) % modulus for a,b in zip(xs,ys)]),
                                    (mul96, [(a*b) % modulus for a,b in zip(xs,ys)])]:
            self.assertEqual(from_limbs(operation(x,y)), expected)
        torch.testing.assert_close(unpack_wire(pack_wire(x)), x)
        self.assertEqual(pack_wire(x).element_size() * pack_wire(x).shape[-1], 12)

    def test_signed_unsigned_boundaries(self):
        values = [-(1 << 63), -1, 0, 1, (1 << 63)-1]
        x = torch.tensor(values, dtype=torch.int64)
        self.assertEqual(from_limbs(limbs_from_int64(x)), [n % (1 << 96) for n in values])
        self.assertEqual(from_limbs(limbs_from_uint64(x)), [n % (1 << 64) for n in values])

    def test_ten_thousand_isqrt_and_boundaries(self):
        rng = random.Random(72)
        values = [rng.randrange(1 << 84) for _ in range(10000)] + [0, 1, (1 << 84)-1]
        for root in [2, (1 << 21)-1, (1 << 32)+1, (1 << 42)-1]:
            values.extend([root*root-1, root*root, root*root+1])
        actual = isqrt_from_limbs(torch.stack([to_limbs(n) for n in values])).tolist()
        self.assertEqual(actual, [math.isqrt(n) for n in values])
        with self.assertRaises(ValueError):
            isqrt_from_limbs(to_limbs(1 << 84))


class ManuscriptAuditTests(unittest.TestCase):
    def test_square_blinding_preserves_two_adic_parity(self):
        # A concrete counterexample to a general statistical-hiding claim.
        def parity(n):
            return ((n & -n).bit_length() - 1) % 2
        for v in [1, 2, 3, 4, 6, 1 << 16, 1 << 17]:
            for a in [(1 << 29), (1 << 29)+1, (1 << 30)-1]:
                self.assertEqual(parity(a*a*v), parity(v))

    def test_shared_scale_recovers_square_from_branch_ratio(self):
        s, mu2, a = 7 * (1 << 16), 1 << 16, (1 << 29) + 5
        y1, y2 = a*a*(s+mu2), a*a*(s+4*mu2)
        # With the SAME square and scale, y2-y1=3*mu²*a² reveals the scale².
        recovered_a2 = (y2-y1) // (3*mu2)
        self.assertEqual(y1 // recovered_a2 - mu2, s)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
