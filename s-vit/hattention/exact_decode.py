"""Decode explicitly opened CrypTen fixed-point integers without legacy rounding."""
import torch


def reveal_exact(value):
    """Reveal to the configured parties and decode using the actual encoder scale."""
    opened = value.reveal()
    raw = opened.tensor() if hasattr(opened, "tensor") else opened
    return raw.to(torch.float64) / value._tensor.encoder.scale
