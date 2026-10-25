"""Inference-only CrypTen MPCTensor adapter; never generates material in forward."""

from .protocol import prepare_g2, ruler_g2


class ExperimentalG2Bridge:
    def __init__(self, base, wide, mu):
        self.base, self.wide, self.mu = base, wide, mu
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("consume pending material before preparing another call")
        self.material = prepare_g2(self.base, self.wide, tuple(shape), self.mu)

    def __call__(self, value):
        from crypten.mpc import MPCTensor
        from crypten.mpc import ptype

        if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic
                or value.encoder._precision_bits != 17):
            raise ValueError("expected CrypTen arithmetic MPCTensor with precision=17")
        if value.requires_grad:
            raise ValueError("bridge supports inference / forward-only tuning, not backward")
        if self.material is None:
            raise RuntimeError("call prepare(shape) before each forward")
        material, self.material = self.material, None
        share = ruler_g2(self.base, self.wide, value.share, material)
        return MPCTensor.from_shares(share, precision=17)
