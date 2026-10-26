"""PUMA-style piecewise-cubic GeLU on the experimental Ring3 runtime.

The arithmetic follows PUMA's published approximation.  This module is a
same-runtime engineering baseline, not the official SecretFlow-SPU PUMA
implementation.  The current Ring3 prototype has no secure less-than
primitive, so interval selection opens Q17 x and performs the branch locally.
Consequently reports must label this path ``puma_style_experimental`` and
must not be interpreted as a security-validated PUMA result.
"""

from dataclasses import dataclass

import torch

from .baseline import PUMA_F0, PUMA_F1, PUMA_T0, PUMA_T1, PUMA_T2
from .protocol import prob_trunc_batch, prob_trunc_material


@dataclass
class PumaMaterial:
    shape: tuple
    runtime: object
    truncation: tuple
    used: bool = False

    def consume(self, base, value):
        if self.used:
            raise RuntimeError("PUMA material already consumed")
        if base is not self.runtime or tuple(value.shape) != self.shape:
            raise ValueError("material belongs to another runtime or tensor shape")
        if value.dtype != torch.int64 or value.device != base.device:
            raise ValueError(f"expected int64 Q17 share on {base.device}")
        self.used = True


def prepare_puma(base, shape):
    """Prepare one-use masks for PUMA's even powers and polynomial products."""
    shape = tuple(shape)
    if any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("shape dimensions must be positive integers")
    # x^2, x^3, x^4, x^6, then three F0 and four F1 coefficient products.
    shifts = (17,) * 11
    masks = tuple(prob_trunc_material(base, shape, shift) for shift in shifts)
    return PumaMaterial(shape, base, masks)


def _q17(value, frac=17):
    return round(float(value) * (1 << frac))


def puma_gelu(base, x_q17, material):
    """Evaluate the PUMA piecewise cubic and return a Q17 secret share.

    The opened Q17 input is used only for experimental branch selection.  The
    polynomial products themselves remain secret-shared in the Ring3 runtime.
    """
    material.consume(base, x_q17)
    square_q17 = prob_trunc_batch(base, (base.square(x_q17),), (17,),
                                  (material.truncation[0],))[0]
    cube_q17 = prob_trunc_batch(base, (base.mul(square_q17, x_q17),), (17,),
                                (material.truncation[1],))[0]
    fourth_q17 = prob_trunc_batch(base, (base.square(square_q17),), (17,),
                                  (material.truncation[2],))[0]
    sixth_q17 = prob_trunc_batch(base, (base.mul(fourth_q17, square_q17),), (17,),
                                 (material.truncation[3],))[0]
    f0_x3, f0_x2, f0_x, f1_x6, f1_x4, f1_x2, f1_x = prob_trunc_batch(
        base,
        (_q17(PUMA_F0[0]) * cube_q17, _q17(PUMA_F0[1]) * square_q17,
         _q17(PUMA_F0[2]) * x_q17, _q17(PUMA_F1[0]) * sixth_q17,
         _q17(PUMA_F1[1]) * fourth_q17, _q17(PUMA_F1[2]) * square_q17,
         _q17(PUMA_F1[3]) * x_q17),
        (17,) * 7, material.truncation[4:11],
    )
    f0 = f0_x3 + f0_x2 + f0_x + base.public(_q17(PUMA_F0[3]))
    f1 = f1_x6 + f1_x4 + f1_x2 + f1_x + base.public(_q17(PUMA_F1[4]))
    # The branch mask is intentionally public in this prototype because Ring3
    # lacks secure LT.
    opened = base.reveal(x_q17)
    negative = opened < round(PUMA_T0 * (1 << 17))
    first_piece = opened < round(PUMA_T1 * (1 << 17))
    central = opened <= round(PUMA_T2 * (1 << 17))
    selected = torch.where(first_piece, f0, f1)
    selected = torch.where(negative, torch.zeros_like(selected), selected)
    selected = torch.where(central, selected, x_q17)
    return selected
