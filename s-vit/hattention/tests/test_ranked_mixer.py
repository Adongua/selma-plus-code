"""Ranking, masking and independent fixed-candidate checks."""

import unittest
from types import SimpleNamespace

import torch

from hattention.ranked_mixer import ranked_paths


def layer(alpha, masks, fixed=True):
    return SimpleNamespace(lsa_alpha=torch.tensor(alpha), allowed_paths=torch.tensor(masks),
                           fixed_rsa_positions=fixed)


class RankedMixerTests(unittest.TestCase):
    def test_global_rank_preserves_rsa_and_has_deterministic_ties(self):
        layers = [layer([.2, .9, .4], [[1, 0, 0], [0, 1, 1], [0, 1, 1]]),
                  layer([.4, .1], [[0, 1, 1], [0, 1, 1]])]
        paths, selected = ranked_paths(layers, 3)
        self.assertEqual(paths, [["rsa", "lsa", "mixer"], ["mixer", "mixer"]])
        self.assertEqual([(item["layer"], item["head"]) for item in selected], [(2, 1), (1, 2), (2, 0)])
        baseline, _ = ranked_paths(layers, 0)
        self.assertEqual(baseline, [["rsa", "lsa", "lsa"], ["lsa", "lsa"]])

    def test_rejects_invalid_alphas_masks_and_count(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ranked_paths([layer([.3], [[0, 1, 1]])], 2)
        with self.assertRaisesRegex(ValueError, "alpha"):
            ranked_paths([layer([float("nan")], [[0, 1, 1]])], 0)
        with self.assertRaisesRegex(ValueError, "mask"):
            ranked_paths([layer([.3], [[0, 1, 0]])], 0)
        with self.assertRaisesRegex(ValueError, "fixed-RSA"):
            ranked_paths([layer([.3], [[0, 1, 1]], fixed=False)], 0)


if __name__ == "__main__":
    unittest.main()
