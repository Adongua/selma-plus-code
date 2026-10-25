"""CrypTen-native MPC GeLU baseline.

This is a framework baseline, not a claim of state of the art. CrypTen's
``erf`` is an MPC Taylor approximation controlled by ``erf_iterations`` in its
configuration (8 by default in the bundled CrypTen).
"""

import math


SECFORMER_BETA = (
    1.25772,
    -0.0299154,
    0.382155,
    -0.0519123,
    0.196033,
    -0.0624557,
    0.118029,
)


def secformer_fourier_gelu_reference(value, threshold=2.5):
    """Plaintext reference for the published seven-term Fourier fit."""
    import torch

    z = value / math.sqrt(2.0)
    fourier = torch.zeros_like(z)
    for harmonic, beta in enumerate(SECFORMER_BETA, start=1):
        fourier = fourier + beta * torch.sin(harmonic * math.pi * z / 10.0)
    erf_approx = torch.where(z < -threshold, -1.0,
                             torch.where(z < threshold, fourier, 1.0))
    return 0.5 * value * (1.0 + erf_approx)


def crypten_erf_gelu(value):
    """Evaluate ``0.5*x*(1+erf(x/sqrt(2)))`` on a CrypTen MPCTensor."""
    from crypten.mpc import MPCTensor, ptype

    if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic):
        raise ValueError("expected an arithmetic CrypTen MPCTensor")
    return value.mul(0.5).mul(
        value.div(math.sqrt(2.0)).erf().add(1.0)
    )


def crypten_quad_gelu(value):
    """MPCFormer's quadratic GeLU approximation: 0.125*x^2+0.25*x+0.5."""
    from crypten.mpc import MPCTensor, ptype

    if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic):
        raise ValueError("expected an arithmetic CrypTen MPCTensor")
    return value.square().mul(0.125).add(value.mul(0.25)).add(0.5)


def crypten_secformer_fourier_gelu(
    value, threshold=2.5, comparison_value=None, comparison_scale=1 << 17
):
    """Privacy-preserving direct-sine reconstruction of SecFormer GeLU.

    The public SecFormer implementation opens ``(x - t)`` before evaluating
    its Fourier lookup, so it is not a strict private MPC baseline.  This
    variant keeps the same seven published Fourier coefficients but evaluates
    every sine and interval predicate directly on CrypTen MPCTensors.  The
    input to the fitted erf is the normalized value ``x / sqrt(2)``; the
    default threshold 2.5 matches the released implementation.  The bundled
    CrypTen comparator only works reliably for precision-zero arithmetic
    shares, so callers may provide an integer-encoded copy of ``x`` as
    ``comparison_value``.  Its public scale is then used for the thresholds.
    """
    from crypten.mpc import MPCTensor, ptype

    if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic):
        raise ValueError("expected an arithmetic CrypTen MPCTensor")

    z = value.div(math.sqrt(2.0))
    if comparison_value is None:
        lower = z.lt(-threshold)
        upper = z.lt(threshold)
    else:
        lower = comparison_value.lt(-threshold * math.sqrt(2.0) * comparison_scale)
        upper = comparison_value.lt(threshold * math.sqrt(2.0) * comparison_scale)
    fourier = None
    for harmonic, beta in enumerate(SECFORMER_BETA, start=1):
        term = z.mul(harmonic * math.pi / 10.0).sin().mul(beta)
        fourier = term if fourier is None else fourier.add(term)
    erf_approx = (-lower).add((upper - lower).mul(fourier)).add(1 - upper)
    return value.mul(0.5).mul(erf_approx.add(1.0))


def crypten_secformer_masked_fourier_gelu(
    value, mask, mask_sin, mask_cos, comparison_value=None,
    comparison_scale=1 << 17, threshold=2.5,
):
    """SecFormer-style masked-opening Fourier reference.

    This mirrors the released implementation: ``value - mask`` is opened,
    the public masked residue is evaluated with ordinary sine/cosine, and
    encrypted ``sin(k*mask)`` / ``cos(k*mask)`` are combined with it.  It is
    an engineering/performance reference, not a standalone security claim.
    """
    from crypten.mpc import MPCTensor, ptype
    import torch

    if (not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic):
        raise ValueError("expected an arithmetic CrypTen MPCTensor")
    delta = torch.remainder((value - mask).get_plain_text(), 2.0 * math.pi)
    coefficients = torch.tensor(SECFORMER_BETA, dtype=delta.dtype, device=delta.device)
    harmonics = torch.arange(1, 8, dtype=delta.dtype, device=delta.device)
    angles = delta.unsqueeze(-1) * (harmonics * math.pi / 10.0)
    p, q = torch.sin(angles), torch.cos(angles)
    fourier = (mask_cos.mul(p) + mask_sin.mul(q)).mul(coefficients).sum(-1)

    if comparison_value is None:
        z = value.div(math.sqrt(2.0))
        lower, upper = z.lt(-threshold), z.lt(threshold)
    else:
        lower = comparison_value.lt(-threshold * math.sqrt(2.0) * comparison_scale)
        upper = comparison_value.lt(threshold * math.sqrt(2.0) * comparison_scale)
    erf_approx = (-lower).add((upper - lower).mul(fourier)).add(1 - upper)
    return value.mul(0.5).mul(erf_approx.add(1.0))
