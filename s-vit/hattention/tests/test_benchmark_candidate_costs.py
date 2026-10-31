import unittest

from hattention.benchmark_candidate_costs import build_summary, extract_candidates, group_pair


class CandidateCostTests(unittest.TestCase):
    def test_selection_validation_and_shared_layer_sum(self):
        report = {"status": "completed_plaintext_not_security_validated", "candidates": [
            {"mixer_count": 0, "head_types": [["rsa", "lsa"], ["rsa", "lsa"]]},
            {"mixer_count": 1, "head_types": [["rsa", "mixer"], ["rsa", "lsa"]]},
        ]}
        candidates = extract_candidates(report, [0, 1], 2)
        self.assertEqual(group_pair(candidates[1][0]), (1, 1))
        def cost(seconds, byte_count, phases):
            return {"online_seconds": seconds, "online_bytes": byte_count, "online_phases": phases}
        rounds = [{(1, 0): cost(3, 20, 4), (1, 1): cost(2, 10, 3)},
                  {(1, 0): cost(5, 20, 4), (1, 1): cost(4, 10, 3)}]
        summary = build_summary(candidates, rounds)
        self.assertEqual(summary[0]["mean_estimate"]["online_seconds"], 8)
        self.assertEqual(summary[1]["mean_estimate"]["online_seconds"], 7)
        self.assertEqual(summary[1]["mean_estimate"]["online_bytes"], 30)
        self.assertEqual(summary[1]["round_estimates"][0]["online_phases"], 7)
        with self.assertRaises(ValueError):
            extract_candidates(report, [24], 2)
        report["candidates"][1]["mixer_count"] = 2
        with self.assertRaises(ValueError):
            extract_candidates(report, [2], 2)


if __name__ == "__main__":
    unittest.main()
