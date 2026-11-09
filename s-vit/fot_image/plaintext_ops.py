"""Explicitly separate plaintext accuracy studies from MPC input contracts."""
import torch
from training.model import PlainOps


class AccuracyPlainOps(PlainOps):
    def __init__(self, params, allow_gelu_out_of_range=False):
        super().__init__(params)
        self.allow_gelu_out_of_range = allow_gelu_out_of_range
        self.gelu_out_of_range_calls = 0
        self.gelu_out_of_range_elements = 0
        self.gelu_max_abs = 0.0

    def check(self, x, bound, site):
        if site.endswith('.gelu_input'):
            if not torch.isfinite(x).all():
                raise ArithmeticError(f'numerical range failure: {site} (nonfinite)')
            max_abs = x.abs().max().item()
            self.gelu_max_abs = max(self.gelu_max_abs, max_abs)
            if max_abs > bound:
                self.gelu_out_of_range_calls += 1
                self.gelu_out_of_range_elements += int((x.abs() > bound).sum().item())
                if self.allow_gelu_out_of_range:
                    return
        return super().check(x, bound, site)

    def range_report(self):
        return {'allow_gelu_out_of_range': self.allow_gelu_out_of_range,
                'gelu_bound': 12.0,
                'gelu_max_abs': self.gelu_max_abs,
                'gelu_out_of_range_calls': self.gelu_out_of_range_calls,
                'gelu_out_of_range_elements': self.gelu_out_of_range_elements,
                'scope': 'all plaintext forward calls, including evaluations and both perturbations; repeated elements counted repeatedly',
                'mpc_validated_for_out_of_range_inputs': False}
