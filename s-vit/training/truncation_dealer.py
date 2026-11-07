"""Local trusted-dealer simulation for fresh Ring3 truncation correlations.

The dealer receives only public shapes and shifts. Its NumPy PRNG and local
IPC are for performance experiments, not a cryptographic implementation.
"""
from contextlib import contextmanager
import importlib

import numpy as np
import torch


def make_shares(shape, shift, rng):
    """Share a uniform 64-bit mask, its logical high bits and its top bit."""
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape) or not 1 <= shift <= 62:
        raise ValueError('invalid truncation material request')
    size = int(np.prod(shape))
    mask = rng.bit_generator.random_raw(size)
    values = (mask, mask >> np.uint64(shift), mask >> np.uint64(63))
    shares = np.empty((3, 3, size), dtype=np.uint64)
    for index, value in enumerate(values):
        shares[0, index] = rng.bit_generator.random_raw(size)
        shares[1, index] = rng.bit_generator.random_raw(size)
        shares[2, index] = value - shares[0, index] - shares[1, index]
    return shares.view(np.int64).reshape((3, 3) + shape)


def serve(connections):
    """Coordinate matching requests and send each rank its own fresh shares."""
    rng = np.random.default_rng()
    try:
        while True:
            requests = [connection.recv() for connection in connections]
            if all(request == ('stop',) for request in requests):
                break
            try:
                if not all(request == requests[0] for request in requests):
                    raise ValueError('party requests do not match')
                shape, shift = requests[0]
                shares = make_shares(shape, shift, rng)
            except Exception as exc:
                for connection in connections:
                    connection.send({'error': str(exc)})
                continue
            for rank, connection in enumerate(connections):
                connection.send({'ok': True})
                connection.send_bytes(shares[rank].tobytes())
    except (EOFError, BrokenPipeError):
        pass
    finally:
        for connection in connections:
            connection.close()


@contextmanager
def trusted_truncations(connection):
    """Replace Ring3 truncation-material generation within this worker only."""
    if connection is None:
        yield
        return
    module_names = ('sgelu.protocol', 'slayernorm.wide_range', 'srelu.protocol',
                    'hattention.protocol', 'hattention.rsa_wide_range')
    modules = [importlib.import_module(name) for name in module_names]
    originals = [module.prob_trunc_material for module in modules]

    def request(base, shape, shift):
        shape = tuple(int(n) for n in shape)
        connection.send((shape, int(shift)))
        reply = connection.recv()
        if 'error' in reply:
            raise RuntimeError('trusted truncation dealer: ' + reply['error'])
        data = connection.recv_bytes()
        expected = 3 * int(np.prod(shape))
        shares = np.frombuffer(data, dtype=np.int64)
        if shares.size != expected:
            raise RuntimeError('trusted truncation dealer returned an invalid payload')
        tensor = torch.from_numpy(shares.copy().reshape((3,) + shape)).to(base.device)
        return tuple(tensor.unbind(0))

    try:
        for module in modules:
            module.prob_trunc_material = request
        yield
    finally:
        for module, original in zip(modules, originals):
            module.prob_trunc_material = original
