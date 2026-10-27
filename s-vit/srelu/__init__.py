"""Standalone secure smooth-ReLU protocol and comparison baselines."""

from .baselines import (
    METHOD_IMPLEMENTATION_CLASS,
    METHOD_MAIN_TABLE_SCOPE,
    METHOD_PROVENANCE,
    METHOD_YEAR,
    METHODS,
    crypten_native_relu,
    effective_signed_bits,
    method_reference,
    quantized_drelu_reference,
)
from .protocol import (
    SReLUMaterial,
    SReLUProfile,
    add_public_half_radius,
    prepare_srelu,
    ruler_srelu,
    ruler_srelu_shifted,
)

__all__ = [
    "METHOD_PROVENANCE",
    "METHOD_IMPLEMENTATION_CLASS",
    "METHOD_MAIN_TABLE_SCOPE",
    "METHOD_YEAR",
    "METHODS",
    "SReLUMaterial",
    "SReLUProfile",
    "add_public_half_radius",
    "crypten_native_relu",
    "effective_signed_bits",
    "method_reference",
    "prepare_srelu",
    "quantized_drelu_reference",
    "ruler_srelu",
    "ruler_srelu_shifted",
]
