"""Verify full forward ordering and public-weight CrypTen operations."""

import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.multiprocessing as mp

from hattention.benchmark_vit_cuda import SecureOperators, audit_public_fixture, load_model
from hattention.config import HAttentionConfig
from hattention.student_checkpoint import import_mpcvit
from hattention.vit_executor import PlainOperators, forward_vit
from hattention.vit_integration import make_fixed_student


ROOT = Path(__file__).resolve().parents[3]


def _crypten_worker(rank, rendezvous):
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=torch.device("cpu"))
    try:
        torch.manual_seed(91)
        x = torch.randn(1, 3, 8, 8) * 0.2
        conv = torch.nn.Conv2d(3, 4, kernel_size=4, stride=4).eval()
        fc = torch.nn.Linear(4, 3).eval()
        cls = torch.randn(1, 1, 4) * 0.1
        encrypted = crypten.cryptensor(x, src=0, precision=17)
        ops = SecureOperators.__new__(SecureOperators)
        actual = ops.conv(encrypted, conv).flatten(2, 3).transpose(-2, -1)
        actual = ops.cat_cls(actual, cls)
        actual = ops.linear(actual, fc).get_plain_text()
        expected = fc(torch.cat([cls, conv(x).flatten(2, 3).transpose(-2, -1)], dim=1))
        if not torch.allclose(actual, expected, atol=0.003, rtol=0):
            raise AssertionError(f"encrypted public layers mismatch: {(actual - expected).abs().max()}")
        # Exercise all three attention paths in a tiny CPU/Gloo graph.
        from crypten.mpc import MPCTensor
        from sgelu.runtime import Ring3, Ring3Wide
        import_mpcvit(ROOT / "mpcvit")
        from src.vit import ViTLite
        torch.manual_seed(17)
        source = ViTLite(img_size=8, kernel_size=4, embedding_dim=12, num_layers=1,
                         num_heads=3, mlp_ratio=2, num_classes=3, dropout=0,
                         attention_dropout=0, stochastic_depth=0)
        config = HAttentionConfig(3, ("rsa", "lsa", "mixer"), patch_grid=(2, 2))
        model = make_fixed_student(source, [config]).eval()
        image = torch.randn(1, 3, 8, 8) * 0.2
        plain = forward_vit(model, image, PlainOperators())
        base = Ring3()
        secure = SecureOperators.__new__(SecureOperators)
        secure.base, secure.wide, secure.mu = base, Ring3Wide(base), 1.4621998268680836
        secure.prepare = lambda fn: fn()
        secure.wrap = lambda share: MPCTensor.from_shares(share, precision=17)
        encrypted = crypten.cryptensor(image, src=0, precision=17)
        out = forward_vit(model, encrypted, secure).get_plain_text()
        if not torch.isfinite(out).all() or (out - plain).abs().max() > 0.5:
            raise AssertionError(f"encrypted full forward mismatch: {out} vs {plain}")
    finally:
        crypten.uninit()


class ViTExecutorTests(unittest.TestCase):
    def test_executor_matches_fixed_model(self):
        checkpoint = ROOT / "mpcvit/mpcvit_checkpoints/mpcvit_tinyimagenet-0.3-kd.pth.tar"
        model, _ = load_model(checkpoint, ROOT / "mpcvit", None)
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected, _ = model(image)
            actual = forward_vit(model, image, PlainOperators())
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))
        audited, observations = audit_public_fixture(model, image)
        self.assertTrue(torch.allclose(expected.double(), audited, atol=1e-6, rtol=1e-6))
        self.assertEqual(len(observations), 37)  # 4 per block and final norm
        self.assertEqual(len(model.classifier.blocks[0].self_attn._forward_pre_hooks), 0)

    def test_crypten_public_linear_layers_on_three_cpu_parties(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_crypten_worker, args=(Path(directory, "store").as_uri(),), nprocs=3, join=True)


if __name__ == "__main__":
    unittest.main()
