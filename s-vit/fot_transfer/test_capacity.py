"""CPU tests for independent prompt capacity and estimator contracts."""
import unittest
import torch
from . import capacity_decoder
from training.core import ZOConfig, direction


class CapacityTests(unittest.TestCase):
    def test_count_output_and_gradients(self):
        theta=capacity_decoder.initial_parameters(123).requires_grad_()
        self.assertEqual(theta.numel(),9007)
        x=torch.randn(2,32,dtype=torch.float64)
        out=capacity_decoder.image_prompt(x,theta,(3,64,64))
        self.assertEqual(tuple(out.shape),(2,3,64,64))
        grad,=torch.autograd.grad(out.square().mean(),theta)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(),0)

    def test_central_difference(self):
        w=capacity_decoder.initial_parameters(123).requires_grad_()
        x=torch.randn(2,32,dtype=torch.float64)
        g=torch.Generator().manual_seed(7)
        u=(2*torch.randint(0,2,(9007,),generator=g).double()-1)
        def objective(z): return capacity_decoder.image_prompt(x,z,(3,64,64)).square().mean()
        value=objective(w); grad,=torch.autograd.grad(value,w); exact=torch.dot(grad,u).item()
        sigma=1e-6
        finite=((objective(w+sigma*u)-objective(w-sigma*u))/(2*sigma)).item()
        self.assertAlmostEqual(finite,exact,delta=1e-6+abs(exact)*1e-4)

    def test_spherical_direction_reproduces_original(self):
        cfg=ZOConfig(seed=20261107)
        generator=torch.Generator().manual_seed(cfg.seed+1000003*4)
        u=torch.randn(4,generator=generator,dtype=torch.float64);u=u/u.norm()
        self.assertTrue(torch.equal(u,direction(4,cfg)))


if __name__ == '__main__': unittest.main()
