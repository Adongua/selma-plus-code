"""Optional one-use CrypTen adapter; forwards raw Q17 shares to Ring3."""

from .protocol import Profile, prepare, layernorm


class ExperimentalLayerNormBridge:
    def __init__(self, base, wide, profile=Profile(), method="fused_ruler"):
        self.base, self.wide = base, wide
        self.profile, self.method = profile, method
        self.material = None

    def prepare(self, shape):
        if self.material is not None:
            raise RuntimeError("consume pending material first")
        self.material = prepare(self.base, self.wide, shape, self.profile, self.method)

    def __call__(self, x, gamma, beta):
        from crypten.mpc import MPCTensor, ptype
        for value in (x, gamma, beta):
            if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic
                    or value.encoder._precision_bits != 17 or value.requires_grad):
                raise ValueError("forward-only arithmetic Q17 MPCTensors required")
        if self.material is None:
            raise RuntimeError("prepare fresh material before each call")
        material, self.material = self.material, None
        share = layernorm(self.base, self.wide, x.share, gamma.share, beta.share, material)
        return MPCTensor.from_shares(share, precision=17)
