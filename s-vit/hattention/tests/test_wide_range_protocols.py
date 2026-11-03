"""Three-party CPU numerical tests for the separate wide-range protocols."""
import tempfile
from pathlib import Path
import math
import random
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from hattention.rsa_wide_range import (WideRSAProfile, prepare_reciprocal, wide_reciprocal,
                                       prepare_wide_rsa, wide_rsa)
from hattention.reference import rsa_attention
from slayernorm.wide_range import WideLayerNormProfile, prepare_wide_layernorm, wide_layernorm
from slayernorm.public_isqrt96 import public_isqrt96
from sgelu.runtime import Ring3, Ring3Wide
from sgelu.wide_range import prepare_wide_gelu, wide_gelu
from sgelu.sgelu import g2
from sgelu.wide import to_limbs


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3)
    try:
        base, wide = Ring3(), None
        wide = Ring3Wide(base)
        denominators = torch.tensor([0.0005, 0.0015, 0.01, 1., 64., 1072., 2048.],
                                    dtype=torch.float64)
        encoded = torch.round(denominators * 2**17).long()
        shared = base.private(encoded, 0)
        reciprocal_material = prepare_reciprocal(base, encoded.shape)
        actual = base.reveal(wide_reciprocal(base, shared, reciprocal_material)).double() / 2**24
        expected = 2**17 / encoded.double()
        relative = ((actual - expected) / expected).abs()
        if relative.max() > 0.02:
            raise AssertionError(f"wide reciprocal relative error {relative.tolist()}")
        try:
            wide_reciprocal(base, shared, reciprocal_material)
        except ValueError:
            pass
        else:
            raise AssertionError("one-use reciprocal material was reused")

        generator = torch.Generator().manual_seed(71)
        x = torch.stack((torch.linspace(-27.5, 27.5, 192),
                         21 + torch.randn(192, generator=generator) * 3,
                         25 + torch.linspace(-0.001, 0.001, 192),
                         torch.full((192,), -20.0),
                         torch.linspace(-63.9, 63.9, 192))).double()
        x_q = torch.round(x * 2**17).long()
        gamma = torch.linspace(0.8, 2.5, 192).double()
        beta = torch.linspace(-0.2, 0.2, 192).double()
        gamma_q, beta_q = torch.round(gamma * 2**17).long(), torch.round(beta * 2**17).long()
        profile = WideLayerNormProfile()
        material = prepare_wide_layernorm(base, wide, tuple(x.shape), profile)
        value = wide_layernorm(base, wide, base.private(x_q, 0), base.private(gamma_q, 0),
                               base.private(beta_q, 0), material)
        actual = base.reveal(value).double() / 2**17
        expected = F.layer_norm(x_q.double() / 2**17, (192,), gamma_q.double() / 2**17,
                                beta_q.double() / 2**17, eps=profile.epsilon)
        error = (actual - expected).abs().amax(dim=1)
        if error.max() > 0.04:
            raise AssertionError(f"wide LayerNorm max error per row: {error.tolist()}")
        try:
            wide_layernorm(base, wide, base.private(x_q, 0), base.private(gamma_q, 0),
                           base.private(beta_q, 0), material)
        except ValueError:
            pass
        else:
            raise AssertionError("one-use LayerNorm material was reused")

        x = torch.tensor([-12., -9.5, -8., -1., 0., 1., 8., 9.5, 12.], dtype=torch.float64)
        encoded = torch.round(x * 2**17).long()
        material = prepare_wide_gelu(base, wide, tuple(encoded.shape))
        actual = base.reveal(wide_gelu(base, wide, base.private(encoded, 0), material)).double() / 2**17
        expected = g2(encoded.double() / 2**17)
        if (actual - expected).abs().max() > 0.01:
            raise AssertionError(f"wide GeLU errors: {(actual - expected).abs().tolist()}")

        # Cover full RSA, not just its reciprocal, at both observed extremes.
        for n, q_value, k_value in ((17, 0.16, -0.5), (100, 0.52, -1.0),
                                    (100, 1.0, None), (73, 14.7, 1.0)):
            shape = (1, 1, n, 1)
            q = torch.full(shape, q_value, dtype=torch.float64)
            k = (torch.linspace(-0.3, -0.8, n, dtype=torch.float64).reshape(shape)
                 if k_value is None else torch.full(shape, k_value, dtype=torch.float64))
            v = torch.linspace(-0.5, 0.9, n, dtype=torch.float64).reshape(shape)
            values = [torch.round(t * 2**17).long() for t in (q, k, v)]
            plain = [t.double() / 2**17 for t in values]
            expected = rsa_attention(*plain, epsilon=round(1e-5 * 2**17) / 2**17)
            shares = [base.private(t, 0) for t in values]
            actual = base.reveal(wide_rsa(base, wide, *shares,
                                 prepare_wide_rsa(base, wide, shape))).double() / 2**17
            if not torch.isfinite(actual).all() or (actual - expected).abs().max() > 0.10:
                raise AssertionError(f"wide RSA n={n} max error {(actual - expected).abs().max()}")
    finally:
        dist.destroy_process_group()


class WideRangeTests(unittest.TestCase):
    def test_public_96_bit_square_root(self):
        randomizer = random.Random(17)
        values = [0, 1, (1 << 84) - 1, 1 << 84, (1 << 96) - 1]
        for _ in range(80):
            value = randomizer.randrange(1 << 96)
            root = math.isqrt(value)
            values.extend((value, root * root, root * root + 1))
        actual = public_isqrt96(torch.stack([to_limbs(value) for value in values])).tolist()
        self.assertEqual(actual, [math.isqrt(value) for value in values])

    def test_profiles_and_epsilon(self):
        profile = WideLayerNormProfile()
        profile.validate((1, 257, 192))
        self.assertLess(abs(profile.effective_epsilon / profile.epsilon - 1), 0.05)
        self.assertEqual(WideRSAProfile().denominator_max, 2048)

    def test_three_party_wide_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_worker, args=(Path(root, "store").as_uri(),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
