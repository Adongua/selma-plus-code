"""CrypTen implementations of secure LayerNorm literature baselines.

The functions in this module preserve ordinary LayerNorm semantics and secret
``gamma`` / ``beta`` parameters.  They are same-framework ports used to make
operator-level comparisons possible; except for ``crypten_native``, timings
must not be presented as measurements of the authors' original runtimes.
"""

import math


CRYPTEN_METHODS = (
    "crypten_native",
    "secformer",
    "puma",
    "sprint",
    "shaft_adapted_3pc",
)

ALL_METHODS = (
    "slayernorm",
    *CRYPTEN_METHODS,
    "newton_engineering",
)

METHOD_YEAR = {
    "slayernorm": "ours",
    "crypten_native": 2021,
    "secformer": 2024,
    "puma": 2025,
    "sprint": 2026,
    "shaft_adapted_3pc": 2025,
    "newton_engineering": "engineering",
}

METHOD_PROVENANCE = {
    "slayernorm": (
        "manuscript fused-variance Ruler protocol through the Ring3 adapter; "
        "Ruler privacy limitations remain"
    ),
    "crypten_native": (
        "CrypTen MPCTensor composition mean -> square/mean -> sqrt -> "
        "reciprocal -> affine; this is the sequential CrypTen baseline "
        "described by PUMA, not a built-in torch LayerNorm shortcut"
    ),
    "secformer": (
        "SecFormer Algorithm 2-style 11-step Goldschmidt inverse square root "
        "ported to CrypTen MPCTensor; this is not an official SecFormer runtime"
    ),
    "puma": (
        "PUMA LayerNorm structure using a direct inverse-square-root followed "
        "by broadcast multiplication, ported from SPU to CrypTen MPCTensor"
    ),
    "sprint": (
        "SPRINT 2026 piecewise inverse-square-root with T=100 and beta=1000, "
        "ported to three-party CrypTen MPCTensor"
    ),
    "shaft_adapted_3pc": (
        "SHAFT 2025 inverse-square-root parameters (8 exponential seed steps, "
        "5 Newton steps) executed in three-party CrypTen; SHAFT itself is 2PC"
    ),
    "newton_engineering": (
        "existing same-Ring3 unfused variance plus 32-step Newton engineering "
        "baseline; not a literature SOTA claim"
    ),
}


def _center_and_variance(value, epsilon):
    mean = value.mean(dim=-1, keepdim=True)
    centered = value - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered, variance.add(epsilon)


def _affine(centered, inverse_std, gamma, beta):
    return centered.mul(inverse_std).mul(gamma).add(beta)


def crypten_native_layernorm(value, gamma, beta, epsilon=1e-5):
    """Sequential sqrt then reciprocal composition used as CrypTen baseline."""
    centered, variance = _center_and_variance(value, epsilon)
    inverse_std = variance.sqrt().reciprocal()
    return _affine(centered, inverse_std, gamma, beta)


def secformer_goldschmidt_rsqrt(value, iterations=11, eta=16.0):
    """SecFormer-style Goldschmidt inverse square root.

    ``q=value/eta`` and ``p=1`` converge to ``q=1`` and
    ``p=sqrt(eta/value)``.  Dividing by ``sqrt(eta)`` yields ``1/sqrt(value)``.
    The operation sequence matches the existing fixed-point Ring3 port.
    """
    q = value.div(eta)
    p = q.mul(0).add(1.0)
    for _ in range(iterations):
        multiplier = q.neg().add(3.0).div(2.0)
        q = q.mul(multiplier.square())
        p = p.mul(multiplier)
    return p.div(math.sqrt(eta))


def secformer_layernorm(value, gamma, beta, epsilon=1e-5):
    centered, variance = _center_and_variance(value, epsilon)
    return _affine(
        centered, secformer_goldschmidt_rsqrt(variance), gamma, beta
    )


def puma_layernorm(value, gamma, beta, epsilon=1e-5):
    """PUMA's direct-rSqrt-and-broadcast LayerNorm structure."""
    centered, variance = _center_and_variance(value, epsilon)
    return _affine(centered, variance.inv_sqrt(), gamma, beta)


def sprint_layernorm(
    value, gamma, beta, epsilon=1e-5, threshold=100.0, beta_scale=1000.0,
):
    """SPRINT's piecewise inverse square root, adapted to CrypTen 3PC.

    Both candidates are evaluated before the secret MUX.  The piecewise path
    is therefore a stability baseline, not an assumption that secret branching
    skips work.
    """
    import crypten

    centered, variance = _center_and_variance(value, epsilon)
    use_unscaled = variance.le(threshold)
    unscaled = variance.inv_sqrt()
    scaled = variance.div(beta_scale).inv_sqrt().div(math.sqrt(beta_scale))
    inverse_std = crypten.where(use_unscaled, unscaled, scaled)
    return _affine(centered, inverse_std, gamma, beta)


def shaft_adapted_3pc_layernorm(value, gamma, beta, epsilon=1e-5):
    """SHAFT rSqrt parameterization executed in CrypTen's 3PC runtime."""
    from crypten.config import cfg

    centered, variance = _center_and_variance(value, epsilon)
    with cfg.temp_override({
        "functions.exp_iterations": 8,
        "functions.sqrt_nr_iters": 5,
    }):
        inverse_std = variance.inv_sqrt()
    return _affine(centered, inverse_std, gamma, beta)


def crypten_layernorm(method, value, gamma, beta, epsilon=1e-5):
    functions = {
        "crypten_native": crypten_native_layernorm,
        "secformer": secformer_layernorm,
        "puma": puma_layernorm,
        "sprint": sprint_layernorm,
        "shaft_adapted_3pc": shaft_adapted_3pc_layernorm,
    }
    if method not in functions:
        raise ValueError(f"not a CrypTen LayerNorm method: {method}")
    return functions[method](value, gamma, beta, epsilon)


def goldschmidt_rsqrt_reference(value, iterations=11, eta=16.0):
    """Plain PyTorch reference for testing the Goldschmidt recurrence."""
    q = value / eta
    p = value.mul(0).add(1.0)
    for _ in range(iterations):
        multiplier = (3.0 - q) / 2.0
        q = q * multiplier.square()
        p = p * multiplier
    return p / math.sqrt(eta)
