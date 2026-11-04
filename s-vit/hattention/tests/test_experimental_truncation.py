"""Public, deterministic wrap fixtures; no privacy claim or production use."""
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    import crypten.communicator as comm
    from crypten.common.util import count_wraps
    from crypten.mpc.primitives import beaver
    from crypten.mpc.primitives.arithmetic import ArithmeticSharedTensor
    from hattention.experimental_crypten_truncation import (batched_truncate, experimental_truncate,
                                                             experimental_truncation)

    crypten.init()
    try:
        values = torch.tensor([1 << 40, -(1 << 40), 0, 123456789, -123456789], dtype=torch.long)
        masks = torch.tensor([(1 << 63) - 1 - (1 << 39), -(1 << 63) + (1 << 39),
                              42, -1000000, 1000000], dtype=torch.long)
        provider = crypten.mpc.get_default_provider()
        original = provider.wrap_rng

        def fixed_mask(size, device=None):
            r = ArithmeticSharedTensor(masks.to(device), precision=0, src=0)
            shares = comm.get().gather(r.share, 0)
            theta = count_wraps(shares) if rank == 0 else torch.zeros_like(masks, device=device)
            return r, ArithmeticSharedTensor(theta, precision=0, src=0)

        provider.wrap_rng = fixed_mask
        try:
            for scale in (192, 1 << 17, 1 << 20):
                x = ArithmeticSharedTensor(values, precision=0, src=0)
                legacy = beaver.truncate(x.clone(), scale).reveal()
                expected = torch.div(values, scale, rounding_mode="trunc")
                legacy_diff = legacy - expected
                if not (legacy_diff[:2].abs() > (1 << 40)).all():
                    raise AssertionError(f"fixture did not force both wraps: {legacy_diff}")
                for function in (experimental_truncate, batched_truncate):
                    corrected_diff = function(x.clone(), scale).reveal() - expected
                    if not (corrected_diff.abs() <= 10).all():
                        raise AssertionError(f"{function.__name__} failed for {scale}: {corrected_diff}")
        finally:
            provider.wrap_rng = original

        minimum, maximum = -(1 << 63), (1 << 63) - 1
        party_shares = [
            torch.tensor([minimum, minimum, maximum, maximum, 0, -1], dtype=torch.long),
            torch.tensor([minimum, -1, maximum, 1, 0, 1], dtype=torch.long),
            torch.tensor([0, 1, 0, -1, 0, 0], dtype=torch.long),
        ]
        # Include underflow exactly to zero and overflow to -2**63.
        raw = party_shares[0] + party_shares[1] + party_shares[2]
        for divisor in (2, 192, 1 << 17, 1 << 20, 1 << 62):
            shared = ArithmeticSharedTensor.from_shares(party_shares[rank].clone(), precision=0)
            expected = torch.div(raw, divisor, rounding_mode="trunc")
            for function in (experimental_truncate, batched_truncate):
                result = function(shared.clone(), divisor).reveal()
                if not ((result - expected).abs() <= 10).all():
                    raise AssertionError(f"{function.__name__} boundary failed for {divisor}: {result}, expected {expected}")

        # Deterministic public full-ring shares exercise different residues and signs.
        generator = torch.Generator().manual_seed(20261104)
        random_shares = [torch.randint(-(1 << 63), (1 << 63) - 1, (256,),
                                       dtype=torch.long, generator=generator) for _ in range(3)]
        random_raw = random_shares[0] + random_shares[1] + random_shares[2]
        for divisor in (192, 1 << 17, 1 << 20):
            expected = torch.div(random_raw, divisor, rounding_mode="trunc")
            for function in (experimental_truncate, batched_truncate):
                shared = ArithmeticSharedTensor.from_shares(random_shares[rank].clone(), precision=0)
                delta = function(shared, divisor).reveal() - expected
                if ((delta > 10) | (delta < -10)).any():
                    raise AssertionError(f"{function.__name__} random-ring failure at divisor {divisor}")

        # Exercise multiplication and an end-to-end softmax attention expression.
        from hattention.benchmark_vit_native_cuda import scaled_standard_attention
        from hattention.exact_decode import reveal_exact
        q = torch.tensor([[[[0.25, -0.5], [0.125, 0.375]]]])
        k = torch.tensor([[[[-0.25, 0.5], [0.75, 0.125]]]])
        v = torch.tensor([[[[0.2, -0.3], [0.4, 0.1]]]])
        secret = [crypten.cryptensor(t, src=0, precision=20) for t in (q, k, v)]
        old_truncate = beaver.truncate
        expected = torch.softmax(q.matmul(k.transpose(-2, -1)) * 2**-0.5, -1).matmul(v)
        for mode in ("experimental", "batched"):
            with experimental_truncation(mode):
                out = reveal_exact(scaled_standard_attention(*secret, 2))
            if beaver.truncate is not old_truncate:
                raise AssertionError("experimental protocol was not restored")
            if not torch.isfinite(out).all() or (out - expected).abs().max() > 0.03:
                raise AssertionError(f"{mode} attention mismatch: {out} vs {expected}")
    finally:
        crypten.uninit()


class ExperimentalTruncationTests(unittest.TestCase):
    def test_forced_positive_and_negative_wraps(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(Path(directory, "store").as_uri(),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
