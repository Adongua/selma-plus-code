"""Fixed-iteration Q24 secret Adam for flattened first-order gradients."""
import math

from .secure import check_secret, negative_bit, precision


class SecretVectorAdam:
    def __init__(self, config, *, allow_numerical_failure=False):
        self.config = config.validate()
        self.allow_numerical_failure = allow_numerical_failure
        self.numerical_failures = []
        self.m = self.v = None
        self.step = 0

    def check(self, value, bound, site, lower=None):
        try:
            check_secret(value, bound, site, lower=lower)
        except ArithmeticError:
            if not self.allow_numerical_failure:
                raise
            self.numerical_failures.append({'step': self.step, 'site': site})

    def update(self, parameters, gradient):
        p = self.config
        gradient = precision(gradient, 24)
        self.step += 1
        self.check(gradient, p.c_bound, 'adam.gradient')
        if self.m is None:
            self.m, self.v = gradient * 0, gradient * 0
        self.m = p.beta1*self.m + (1-p.beta1)*gradient
        self.v = p.beta2*self.v + (1-p.beta2)*gradient.square()
        mh = self.m / (1-p.beta1**self.step)
        vh = self.v / (1-p.beta2**self.step)
        # The sign is Q0; multiplying by it keeps Q24 without a truncation,
        # avoiding a fresh negative rounding error on exact zero entries.
        vh = vh * (1-negative_bit(vh))
        self.check(vh, p.c_bound**2 + .01, 'adam.v_hat', lower=0)
        hi, lo = vh*0 + math.sqrt(p.c_bound**2+.01), vh*0
        for _ in range(p.sqrt_iters):
            middle = (lo+hi)*.5
            choose_low = precision(1-negative_bit(vh-middle*middle), 24)
            lo = choose_low*middle + (1-choose_low)*lo
            hi = choose_low*hi + (1-choose_low)*middle
        denominator = (lo+hi)*.5 + p.epsilon
        inverse = denominator*0 + 1/(math.sqrt(p.c_bound**2+.01)+p.epsilon)
        for _ in range(p.reciprocal_iters):
            inverse = inverse*(2-denominator*inverse)
        self.check(denominator*inverse-1, .02, 'adam.reciprocal_residual')
        bits = parameters._tensor.encoder.scale.bit_length() - 1
        return parameters - precision(p.lr*mh*inverse, bits)
