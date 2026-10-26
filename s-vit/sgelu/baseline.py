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


PUMA_T0 = -4.0
PUMA_T1 = -1.95
PUMA_T2 = 3.0
PUMA_F0 = (-0.011034134030615728, -0.11807612951181953,
           -0.42226581151983866, -0.5054031199708174)
PUMA_F1 = (0.0018067462606141187, -0.037688200365904236,
           0.3603292692789629, 0.5, 0.008526321541038084)


# SEAF (USENIX Security 2025) reduces GeLU to the even function
# g(|x|) = .5 |x| erf(|x| / sqrt(2)) and uses a 64-entry lookup over
# [0, 4).  Its published high-accuracy configuration has fractional precisions
# (f_a, f_d) = (7, 12).  The paper's optimized protocol is 2PC and relies on a
# bespoke secret LUT plus Trun-Eq, neither of which is a CrypTen primitive.
#
# The implementation below is consequently named ``seaf_pwl_3pc`` rather than
# ``SEAF``: it preserves the public approximation structure and quantization,
# but evaluates the secret interval selection with batched CrypTen comparisons.
# The coefficients are deterministically regenerated from the exact GeLU
# formula at segment midpoints; they are not claimed to be the authors' private
# lookup-file bytes.
SEAF_PWL_SEGMENTS = 64
SEAF_PWL_BOUND = 4.0
SEAF_PWL_SLOPE_FRACTION_BITS = 7
SEAF_PWL_INTERCEPT_FRACTION_BITS = 12


def seaf_pwl_table(*, device=None, dtype=None):
    """Return the public 64-segment, SEAF-style table for ``g(|x|)``.

    Each segment uses its midpoint tangent and then independently quantizes
    the slope and intercept to the published SEAF1 fractional precisions.
    ``thresholds`` contains the exclusive upper endpoint of every segment.
    """
    import torch

    if dtype is None:
        dtype = torch.float64
    width = SEAF_PWL_BOUND / SEAF_PWL_SEGMENTS
    slopes, intercepts = [], []
    for index in range(SEAF_PWL_SEGMENTS):
        midpoint = (index + 0.5) * width
        erf_value = math.erf(midpoint / math.sqrt(2.0))
        g_value = 0.5 * midpoint * erf_value
        slope = (0.5 * erf_value
                 + midpoint * math.exp(-0.5 * midpoint * midpoint)
                 / math.sqrt(2.0 * math.pi))
        intercept = g_value - slope * midpoint
        slopes.append(round(slope * (1 << SEAF_PWL_SLOPE_FRACTION_BITS))
                      / (1 << SEAF_PWL_SLOPE_FRACTION_BITS))
        intercepts.append(round(intercept * (1 << SEAF_PWL_INTERCEPT_FRACTION_BITS))
                          / (1 << SEAF_PWL_INTERCEPT_FRACTION_BITS))
    return {
        "slopes": torch.tensor(slopes, dtype=dtype, device=device),
        "intercepts": torch.tensor(intercepts, dtype=dtype, device=device),
        "thresholds": torch.arange(
            1, SEAF_PWL_SEGMENTS + 1, dtype=dtype, device=device
        ) * width,
    }


def seaf_pwl_3pc_gelu_reference(value, table=None):
    """Plaintext reference of the reproducible SEAF-style 3PC PWL port."""
    import torch

    if table is None:
        table = seaf_pwl_table(device=value.device, dtype=value.dtype)
    absolute = value.abs()
    indices = torch.floor(
        absolute * (SEAF_PWL_SEGMENTS / SEAF_PWL_BOUND)
    ).long().clamp(max=SEAF_PWL_SEGMENTS - 1)
    inside = table["slopes"][indices] * absolute + table["intercepts"][indices]
    g_value = torch.where(
        absolute < SEAF_PWL_BOUND, inside, absolute.mul(0.5)
    )
    return value.mul(0.5).add(g_value)


def puma_gelu_reference(value):
    """Plaintext reference for PUMA's published piecewise cubic GeLU.

    This is the approximation from PUMA Eq. (3), not the exact GELU.  Keeping
    this function separate makes the microbenchmark error target explicit.
    """
    import torch

    x = value
    x2 = x * x
    f0 = PUMA_F0[0] * x**3 + PUMA_F0[1] * x2 + PUMA_F0[2] * x + PUMA_F0[3]
    f1 = (PUMA_F1[0] * x**6 + PUMA_F1[1] * x**4 + PUMA_F1[2] * x2
          + PUMA_F1[3] * x + PUMA_F1[4])
    return torch.where(x < PUMA_T0, torch.zeros_like(x),
                       torch.where(x < PUMA_T1, f0,
                                   torch.where(x <= PUMA_T2, f1, x)))


def crypten_puma_gelu(value, comparison_value=None, comparison_scale=1 << 17):
    """CrypTen-native implementation of PUMA's piecewise polynomial GeLU.

    This is a same-framework reimplementation of PUMA's published
    approximation, not the official SecretFlow-SPU PUMA runtime.  All powers,
    polynomial evaluation, and interval masks stay in CrypTen.  When
    ``comparison_value`` is supplied it is an encrypted precision-zero copy
    of Q17 ``x``; this avoids asking CrypTen's comparator to interpret a
    fractional fixed-point value.
    """
    from crypten.mpc import MPCTensor, ptype

    if not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic:
        raise ValueError("expected an arithmetic CrypTen MPCTensor")

    x2 = value.square()
    x3 = x2.mul(value)
    x4 = x2.square()
    x6 = x4.mul(x2)
    f0 = (x3.mul(PUMA_F0[0]).add(x2.mul(PUMA_F0[1]))
          .add(value.mul(PUMA_F0[2])).add(PUMA_F0[3]))
    f1 = (x6.mul(PUMA_F1[0]).add(x4.mul(PUMA_F1[1]))
          .add(x2.mul(PUMA_F1[2])).add(value.mul(PUMA_F1[3]))
          .add(PUMA_F1[4]))

    if comparison_value is None:
        lower = value.lt(PUMA_T0)
        middle = value.lt(PUMA_T1)
        upper = value.lt(PUMA_T2)
    else:
        if (not isinstance(comparison_value, MPCTensor)
                or comparison_value.ptype != ptype.arithmetic):
            raise ValueError("comparison_value must be an arithmetic MPCTensor")
        lower = comparison_value.lt(round(PUMA_T0 * comparison_scale))
        middle = comparison_value.lt(round(PUMA_T1 * comparison_scale))
        upper = comparison_value.lt(round(PUMA_T2 * comparison_scale))

    # Masks are mutually exclusive: [T0,T1), [T1,T2), and [T2,+inf).
    first = middle - lower
    second = upper - middle
    above = 1 - upper
    return first.mul(f0).add(second.mul(f1)).add(above.mul(value))


def crypten_seaf_pwl_3pc_gelu(
    value, comparison_value=None, comparison_scale=1 << 17, table=None,
):
    """Evaluate a SEAF-style 64-segment PWL GeLU using CrypTen 3PC.

    This is deliberately *not* presented as the official SEAF protocol.
    SEAF's 2025 construction is a 2PC scheme with a specialised secret LUT
    and Trun-Eq.  CrypTen has neither primitive, so this adaptation uses one
    vectorized comparison over all 64 public breakpoints and arithmetic
    selection.  It is useful as a same-runtime, recent-method structural
    baseline, but its communication cost must not be attributed to SEAF.

    ``comparison_value`` is a precision-zero, Q17 integer encoding of
    ``value``.  Supplying it avoids CrypTen interpreting fractional shares in
    the comparator; this is the same convention used by the PUMA baseline.
    """
    import crypten
    import torch
    from crypten.mpc import MPCTensor, ptype

    if not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic:
        raise ValueError("expected an arithmetic CrypTen MPCTensor")
    if comparison_value is None:
        raise ValueError("seaf_pwl_3pc requires a precision-zero comparison share")
    if (not isinstance(comparison_value, MPCTensor)
            or comparison_value.ptype != ptype.arithmetic
            or comparison_value.encoder._precision_bits != 0):
        raise ValueError("comparison_value must be a precision-zero arithmetic MPCTensor")
    if table is None:
        table = seaf_pwl_table(device=value.device, dtype=torch.float64)

    shape = tuple(value.size())
    view_shape = (SEAF_PWL_SEGMENTS,) + (1,) * len(shape)
    expanded_shape = (SEAF_PWL_SEGMENTS,) + shape

    # The residual g is even.  Obtain |x| without revealing the sign.  Since
    # this is a 3PC port, CrypTen's generic comparison and multiplication are
    # intentionally included in the reported cost.
    negative = comparison_value.lt(0)
    sign = 1 - negative.mul(2)
    absolute = value.mul(sign)
    comparison_absolute = comparison_value.mul(sign)

    thresholds = (table["thresholds"] * comparison_scale).round().long()
    upper = comparison_absolute.unsqueeze(0).expand(expanded_shape).lt(
        thresholds.view(view_shape)
    )
    masks = crypten.cat(
        [upper[:1], upper[1:].sub(upper[:-1])], dim=0
    )
    lines = absolute.unsqueeze(0).mul(
        table["slopes"].view(view_shape)
    ).add(table["intercepts"].view(view_shape))
    inside = masks.mul(lines).sum(0)
    tail = 1 - upper[-1]
    g_value = inside.add(tail.mul(absolute).mul(0.5))
    return value.mul(0.5).add(g_value)


# EAST (IEEE TSC 2025) gives its OPPE evaluator configuration for GeLU as
# m=8 segments and degree 3, but does not publish the complete set of fitted
# GeLU coefficients.  The functions below therefore implement an explicitly
# labelled *EAST-style* OPPE 3PC port: the public cubic coefficients are
# deterministically regenerated from four Chebyshev nodes per segment.
EAST_OPPE_SEGMENTS = 8
EAST_OPPE_DEGREE = 3
EAST_OPPE_BOUND = 4.0


def _east_cubic_coefficients(lower, upper):
    """Return global-basis cubic interpolant coefficients for exact GeLU."""
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    nodes = [
        center + radius * math.cos((2 * index + 1) * math.pi / 8.0)
        for index in range(EAST_OPPE_DEGREE + 1)
    ]
    coefficients = [0.0] * (EAST_OPPE_DEGREE + 1)
    for index, node in enumerate(nodes):
        basis = [1.0]
        denominator = 1.0
        for other_index, other_node in enumerate(nodes):
            if other_index == index:
                continue
            next_basis = [0.0] * (len(basis) + 1)
            for power, coefficient in enumerate(basis):
                next_basis[power] -= coefficient * other_node
                next_basis[power + 1] += coefficient
            basis = next_basis
            denominator *= node - other_node
        gelu = 0.5 * node * (1.0 + math.erf(node / math.sqrt(2.0)))
        for power, coefficient in enumerate(basis):
            coefficients[power] += gelu * coefficient / denominator
    return coefficients


def east_oppe_table(*, device=None, dtype=None):
    """Return the public ``m=8``, degree-three EAST-style OPPE table."""
    import torch

    if dtype is None:
        dtype = torch.float64
    width = 2.0 * EAST_OPPE_BOUND / EAST_OPPE_SEGMENTS
    coefficients = [
        _east_cubic_coefficients(
            -EAST_OPPE_BOUND + index * width,
            -EAST_OPPE_BOUND + (index + 1) * width,
        )
        for index in range(EAST_OPPE_SEGMENTS)
    ]
    return {
        "coefficients": torch.tensor(coefficients, dtype=dtype, device=device),
        "thresholds": torch.linspace(
            -EAST_OPPE_BOUND,
            EAST_OPPE_BOUND,
            EAST_OPPE_SEGMENTS + 1,
            dtype=dtype,
            device=device,
        ),
    }


def east_oppe_3pc_gelu_reference(value, table=None):
    """Plaintext reference for the deterministic EAST-style 3PC port."""
    import torch

    if table is None:
        table = east_oppe_table(device=value.device, dtype=value.dtype)
    width = 2.0 * EAST_OPPE_BOUND / EAST_OPPE_SEGMENTS
    indices = torch.floor((value + EAST_OPPE_BOUND) / width).long().clamp(
        min=0, max=EAST_OPPE_SEGMENTS - 1
    )
    coefficients = table["coefficients"][indices]
    inside = coefficients[..., 0]
    for power in range(1, EAST_OPPE_DEGREE + 1):
        inside = inside + coefficients[..., power] * value.pow(power)
    return torch.where(
        value < -EAST_OPPE_BOUND,
        torch.zeros_like(value),
        torch.where(value < EAST_OPPE_BOUND, inside, value),
    )


def crypten_east_oppe_3pc_gelu(
    value, comparison_value=None, comparison_scale=1 << 17, table=None,
):
    """Evaluate a CrypTen 3PC port of EAST's ``m=8`` cubic OPPE structure.

    The original EAST protocol is a 2PC construction based on OT-backed MUX.
    CrypTen provides neither that MUX primitive nor its unpublished GELU fit;
    this function evaluates all eight deterministic cubic candidates and uses
    batched CrypTen comparisons for oblivious selection.  It is therefore an
    EAST-style structural baseline, not a report of official EAST performance.
    """
    import crypten
    import torch
    from crypten.mpc import MPCTensor, ptype

    if not isinstance(value, MPCTensor) or value.ptype != ptype.arithmetic:
        raise ValueError("expected an arithmetic CrypTen MPCTensor")
    if comparison_value is None:
        raise ValueError("east_oppe_3pc requires a precision-zero comparison share")
    if (not isinstance(comparison_value, MPCTensor)
            or comparison_value.ptype != ptype.arithmetic
            or comparison_value.encoder._precision_bits != 0):
        raise ValueError("comparison_value must be a precision-zero arithmetic MPCTensor")
    if table is None:
        table = east_oppe_table(device=value.device, dtype=torch.float64)

    shape = tuple(value.size())
    view_shape = (EAST_OPPE_SEGMENTS,) + (1,) * len(shape)
    expanded_shape = (EAST_OPPE_SEGMENTS + 1,) + shape
    thresholds = (table["thresholds"] * comparison_scale).round().long()
    upper = comparison_value.unsqueeze(0).expand(expanded_shape).lt(
        thresholds.view((EAST_OPPE_SEGMENTS + 1,) + (1,) * len(shape))
    )
    # ``upper[i+1] - upper[i]`` selects interval i.  Values below -4 map to
    # zero and values at/above 4 map to x, matching the standard GeLU tails.
    masks = upper[1:].sub(upper[:-1])

    coefficients = table["coefficients"]
    x2 = value.square()
    x3 = x2.mul(value)
    candidates = (
        value.unsqueeze(0).mul(coefficients[:, 1].view(view_shape))
        .add(coefficients[:, 0].view(view_shape))
        .add(x2.unsqueeze(0).mul(coefficients[:, 2].view(view_shape)))
        .add(x3.unsqueeze(0).mul(coefficients[:, 3].view(view_shape)))
    )
    inside = masks.mul(candidates).sum(0)
    return inside.add((1 - upper[-1]).mul(value))


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
    # The Fourier fit approximates erf(z), where z=x/sqrt(2).  The mask and
    # its sine/cosine tables are sampled in the z-domain, so the opened
    # residue must be z-mask rather than x-mask.
    z = value.div(math.sqrt(2.0))
    # Do not reduce delta modulo 2*pi.  The fitted basis uses
    # sin(k*pi*z/10), whose common period in z is 20, not 2*pi.
    delta = (z - mask).get_plain_text()
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
