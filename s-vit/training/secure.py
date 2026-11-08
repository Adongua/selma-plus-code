"""Secret-weight forward, selective output and bounded fixed-iteration Adam.

Only numerical failure bits are public. Ruler and carry-save remain research
protocols, not security-validated implementations.
"""
import math
import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext


def open_to(value, dst):
    """Exact integer decoding; no all-party output reconstruction."""
    opened = value.reveal(dst=dst)
    if dist.get_rank() != dst:
        return None
    raw = opened.tensor() if hasattr(opened, 'tensor') else opened
    return raw.double() / value._tensor.encoder.scale


def raw_share(value):
    share = value.share
    return share.tensor() if hasattr(share, 'tensor') else share


def from_share(value, bits):
    from crypten.mpc import MPCTensor
    from crypten.cuda import CUDALongTensor
    return MPCTensor.from_shares(CUDALongTensor(value) if value.is_cuda else value, precision=bits)


def precision(value, bits):
    from crypten.encoder import FixedPointEncoder
    out = value.clone()
    out._tensor.encode_(FixedPointEncoder(precision_bits=bits))
    return out


def negative_bit(value):
    """Full-ring sign bit, independent of CrypTen's global precision setting.

    The bundled _ltz converts using the global precision and can zero the bit
    when a Q24 input is compared under global Q17. Convert raw sign as Q0.
    """
    from crypten.mpc.primitives.converters import convert
    from crypten.mpc.ptype import ptype
    from crypten.encoder import FixedPointEncoder
    bits = convert(value._tensor, ptype.binary)
    sign = (bits >> 63) & 1
    sign.encoder = FixedPointEncoder(precision_bits=0)
    result = convert(sign, ptype.arithmetic, precision=0, bits=1)
    raw = result.share.tensor() if hasattr(result.share, 'tensor') else result.share
    return from_share(raw, 0)


def project_nonnegative(value):
    """Secret projection with a Q0 mask, avoiding post-projection truncation."""
    return value * (1-negative_bit(value))


def check_secret(value, bound, site, lower=None):
    # Explicit public abort bit. No secret value, loss or optimizer state opened.
    lo = -bound if lower is None else lower
    violations = (negative_bit(bound-value) + negative_bit(value-lo)).sum()
    flag = negative_bit(-violations)
    opened = flag.get_plain_text()
    if bool(opened.item()):
        raise ArithmeticError(f"public numerical failure flag: {site}")


class SecretOps:
    def __init__(self, params, device, trace=None, check_forward_ranges=False):
        if device.type == 'cpu':
            from sgelu.runtime import Ring3, Ring3Wide
            self.base = Ring3()
            self.wide = Ring3Wide(self.base)
        else:
            from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
            self.base = CudaRing3(device=device, reveal_mode='p2p')
            self.wide = CudaRing3Wide(self.base)
        self.params = params
        self.trace = trace
        self.check_forward_ranges = check_forward_ranges
        self.step_benchmark = None

    @contextmanager
    def span(self, category, site):
        trace = self.trace.span(category, site) if self.trace else nullcontext()
        measured = (self.step_benchmark.preparation() if self.step_benchmark and category == 'ring3_prepare'
                    else nullcontext())
        with trace, measured:
            yield

    def linear(self, x, name):
        out = x.matmul(self.params[name+'.weight'].t())
        return out + self.params[name+'.bias'] if name+'.bias' in self.params else out

    def conv(self, x, spec):
        out = x.conv2d(self.params['conv.weight'], stride=tuple(spec['stride']), padding=tuple(spec['padding']))
        return out + self.params['conv.bias'].reshape(1, -1, 1, 1) if 'conv.bias' in self.params else out

    def cat(self, tensors, dim):
        import crypten
        return crypten.cat(tensors, dim=dim)

    def check(self, x, bound, site):
        if not self.check_forward_ranges:
            return
        with self.span('range_check', site):
            check_secret(x, bound, site)

    def norm(self, x, name, epsilon):
        from slayernorm.wide_range import WideLayerNormProfile, prepare_wide_layernorm, wide_layernorm
        for key in ('weight', 'bias'):
            self.check(self.params[name+'.'+key], 4, name+'.'+key)
        with self.span('ring3_prepare', name+'.layernorm'):
            mat = prepare_wide_layernorm(self.base, self.wide, tuple(x.shape), WideLayerNormProfile(epsilon=epsilon))
        out = wide_layernorm(self.base, self.wide, raw_share(x), raw_share(self.params[name+'.weight']),
                             raw_share(self.params[name+'.bias']), mat)
        return from_share(out, 17)

    def gelu(self, x):
        from sgelu.wide_range import prepare_wide_gelu, wide_gelu
        with self.span('ring3_prepare', 'gelu'):
            mat = prepare_wide_gelu(self.base, self.wide, tuple(x.shape))
        return from_share(wide_gelu(self.base, self.wide, raw_share(x), mat), 17)

    def attention(self, q, k, v, block, spec):
        # Keep RSA stages explicit to check score and denominator contracts
        # without opening them. Linear/Mixer implementation matches wide hybrid.
        import crypten
        from hattention.rsa_wide_range import prepare_wide_rsa, wide_reciprocal
        from hattention.protocol import secure_matmul, FixedPointProfile, prepare_lsa, secure_lsa
        from hattention.reference import mixer_attention
        from srelu.protocol import ruler_srelu
        from sgelu.protocol import prob_trunc
        output = [None] * len(block['paths'])
        for path in ('rsa', 'lsa', 'mixer'):
            ids = [i for i, p in enumerate(block['paths']) if p == path]
            if not ids:
                continue
            qs, ks, vs = [crypten.stack([value[:, i] for i in ids], dim=1) for value in (q, k, v)]
            qr, kr, vr = [raw_share(value) for value in (qs, ks, vs)]
            shape = tuple(qr.shape)
            if path == 'rsa':
                with self.span('ring3_prepare', 'rsa'):
                    mat = prepare_wide_rsa(self.base, self.wide, shape)
                mat.used = True
                def trunc(name, value):
                    shift, correlation = mat.truncations[name]
                    return prob_trunc(self.base, value, shift, correlation)
                score = trunc('score', secure_matmul(self.base, qr, kr.transpose(-2, -1)))
                score = trunc('score_scale', score * round(shape[-1]**-.5 * 2**20))
                self.check(from_share(score, 17), 16 - 2**-17, 'rsa.score')
                positive = ruler_srelu(self.base, self.wide, score, mat.srelu)
                denominator = positive.sum(-1, keepdim=True) + self.base.public(round(mat.profile.epsilon*2**17))
                if self.check_forward_ranges:
                    with self.span('range_check', 'rsa.denominator'):
                        check_secret(from_share(denominator, 17), 2048, 'rsa.denominator', lower=.0005)
                inv = wide_reciprocal(self.base, denominator, mat.reciprocal, mat.profile)
                weighted = trunc('weight', self.base.mul(positive, inv))
                group = trunc('output', secure_matmul(self.base, weighted, vr))
            elif path == 'lsa':
                with self.span('ring3_prepare', 'lsa'):
                    mat = prepare_lsa(self.base, shape, shape[-1]**-.5, FixedPointProfile(input_bound=16))
                group = secure_lsa(self.base, qr, kr, vr, mat)
            else:
                group = mixer_attention(vr, tuple(spec['grid']), True)
            wrapped = from_share(group, 17)
            for j, original in enumerate(ids):
                output[original] = wrapped[:, j]
        return crypten.stack(output, dim=1)


class SecretAdam:
    """Q24 latent state. No plaintext delta, m, v or z; epsilon OUTSIDE sqrt."""
    def __init__(self, config, device):
        import crypten
        self.config = config.validate()
        self.z = crypten.cryptensor(torch.zeros(config.rank, device=device), src=0, precision=24)
        self.m, self.v = self.z.clone(), self.z.clone()
        self.step = 0

    def update(self, delta, u):
        p = self.config
        c = delta * (p.rank / (2*p.sigma))
        check_secret(c, p.c_bound, 'adam.c')
        self.step += 1
        self.m = p.beta1*self.m + (1-p.beta1)*(c*u)
        # One SECRET scalar square, public per-coordinate u^2.
        self.v = p.beta2*self.v + (1-p.beta2)*(c*c)*u.square()
        mh = self.m / (1-p.beta1**self.step)
        vh = self.v / (1-p.beta2**self.step)
        # The sign mask is Q0. Multiplying Q24 by Q0 preserves precision without
        # truncation, so a projected zero cannot acquire a new negative value.
        vh = project_nonnegative(vh)
        check_secret(vh, p.c_bound**2 + .01, 'adam.v_hat', lower=0)
        hi = vh*0 + math.sqrt(p.c_bound**2 + .01)
        lo = vh*0
        for _ in range(p.sqrt_iters):
            middle = (lo+hi)*.5
            choose_low = precision(1-negative_bit(vh-middle*middle), 24)
            lo = choose_low*middle + (1-choose_low)*lo
            hi = choose_low*hi + (1-choose_low)*middle
        denominator = (lo+hi)*.5 + p.epsilon
        inverse = denominator*0 + 1/(math.sqrt(p.c_bound**2+.01)+p.epsilon)
        for _ in range(p.reciprocal_iters):
            inverse = inverse*(2-denominator*inverse)
        check_secret(denominator*inverse-1, .02, 'adam.reciprocal_residual')
        self.z = self.z - p.lr*mh*inverse
        check_secret(self.z, p.z_bound, 'adam.z')
        return self.z

    def state(self):
        return {'z_share': raw_share(self.z).cpu(), 'm_share': raw_share(self.m).cpu(),
                'v_share': raw_share(self.v).cpu(), 'precision': 24, 'step': self.step}
