import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hattention import benchmark_cuda


class BackendSetupTests(unittest.TestCase):
    def test_process_group_selects_device_only_for_nccl(self):
        device = torch.device("cuda:1")
        with patch.object(benchmark_cuda.dist, "init_process_group") as init:
            benchmark_cuda._init_process_group(1, "file:///store", device, "gloo")
            args, kwargs = init.call_args
            self.assertEqual(args, ("gloo",))
            self.assertNotIn("device_id", kwargs)
            self.assertEqual(kwargs["rank"], 1)
            self.assertEqual(kwargs["world_size"], 3)

            benchmark_cuda._init_process_group(1, "file:///store", device, "nccl")
            args, kwargs = init.call_args
            self.assertEqual(args, ("nccl",))
            self.assertEqual(kwargs["device_id"], device)

    def test_barrier_selects_device_only_for_nccl(self):
        device = torch.device("cuda:1")
        with patch.object(benchmark_cuda.dist, "barrier") as barrier:
            benchmark_cuda._barrier("gloo", device)
            barrier.assert_called_once_with()
            barrier.reset_mock()
            benchmark_cuda._barrier("nccl", device)
            barrier.assert_called_once_with(device_ids=[1])

    def test_gloo_measure_gathers_cpu_metrics_and_excludes_gather(self):
        base = SimpleNamespace(
            backend="gloo", device=torch.device("cpu"), sent_bytes=7, phases=3
        )

        def operation():
            base.sent_bytes += 12
            base.phases += 2
            return "result"

        def gather(outputs, local):
            self.assertEqual(local.device.type, "cpu")
            for output in outputs:
                output.copy_(local)

        with patch.object(benchmark_cuda.dist, "barrier") as barrier, \
             patch.object(benchmark_cuda.dist, "all_gather", side_effect=gather), \
             patch.object(benchmark_cuda.torch.cuda, "synchronize"):
            result, metrics = benchmark_cuda._measure(base, operation)

        self.assertEqual(result, "result")
        self.assertEqual(metrics["sent_bytes_sum"], 36)
        self.assertEqual(metrics["exchange_phases_per_rank"], [2, 2, 2])
        self.assertGreaterEqual(metrics["seconds_max"], 0.0)
        barrier.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
