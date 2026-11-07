import multiprocessing as mp
import threading
import unittest

import numpy as np

from training.truncation_dealer import make_shares, serve


class DealerTests(unittest.TestCase):
    def test_correlations_and_freshness(self):
        rng = np.random.default_rng(2026)
        shares = make_shares((2, 31), 17, rng).view(np.uint64)
        reconstructed = shares.sum(axis=0, dtype=np.uint64)
        np.testing.assert_array_equal(reconstructed[1], reconstructed[0] >> np.uint64(17))
        np.testing.assert_array_equal(reconstructed[2], reconstructed[0] >> np.uint64(63))
        fresh = make_shares((2, 31), 17, rng).view(np.uint64).sum(axis=0, dtype=np.uint64)
        self.assertFalse(np.array_equal(reconstructed[0], fresh[0]))

    def test_three_party_service(self):
        pairs = [mp.Pipe() for _ in range(3)]
        servers, clients = zip(*pairs)
        thread = threading.Thread(target=serve, args=(servers,), daemon=True)
        thread.start()
        try:
            for client in clients:
                client.send(((4, 7), 20))
            shares = []
            for client in clients:
                self.assertTrue(client.recv()['ok'])
                shares.append(np.frombuffer(client.recv_bytes(), dtype=np.uint64).reshape(3, 4, 7))
            reconstructed = np.stack(shares).sum(axis=0, dtype=np.uint64)
            np.testing.assert_array_equal(reconstructed[1], reconstructed[0] >> np.uint64(20))
            np.testing.assert_array_equal(reconstructed[2], reconstructed[0] >> np.uint64(63))
            for client in clients:
                client.send(('stop',))
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        finally:
            for client in clients:
                client.close()


if __name__ == '__main__':
    unittest.main()
