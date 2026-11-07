"""Independent local-process trusted optimizer, outside the 3-party group.

Queue transport is a local simulation, NOT an authenticated remote service.
"""
import secrets
import numpy as np
import torch
from .core import PlainAdam, direction


def pack(value):
    return value.detach().cpu().contiguous().numpy().astype('<i8').tobytes()


def unpack(value, shape):
    return torch.from_numpy(np.frombuffer(value, dtype='<i8').copy()).reshape(shape)


def split_q24(value):
    raw = torch.round(value*2**24).long()
    a = unpack(secrets.token_bytes(raw.numel()*8), raw.shape)
    b = unpack(secrets.token_bytes(raw.numel()*8), raw.shape)
    return [pack(a), pack(b), pack(raw-a-b)]


def serve(config, steps, incoming, outgoing, private_path):
    try:
        adam = PlainAdam(config)
        def deliver():
            for queue, share in zip(outgoing, split_q24(adam.z)):
                queue.put({'share': share})
        deliver()
        for step in range(1, steps+1):
            shares = {}
            for _ in range(3):
                party, actual_step, payload = incoming.get(timeout=7200)
                if actual_step != step or party in shares or party not in (0, 1, 2):
                    raise ValueError("trusted service message out of sequence")
                shares[party] = unpack(payload, ())
            delta = sum(shares.values()).double().item()/2**24
            adam.update(delta, direction(step, config))
            deliver()
        torch.save(adam.state(), private_path)
    except Exception as error:
        for queue in outgoing:
            queue.put({'error': str(error)})
        raise


def receive(queue, config, device):
    from .secure import from_share
    message = queue.get(timeout=7200)
    if 'error' in message:
        raise RuntimeError('trusted optimizer: '+message['error'])
    return from_share(unpack(message['share'], (config.rank,)).to(device), 24)
