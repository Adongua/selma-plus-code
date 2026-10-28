"""Public constants for exact evaluation of a quantized sign criterion."""


def quantized_sign_offset(input_fraction_bits: int, comparison_fraction_bits: int) -> float:
    """Offset making round-to-even quantized x >= 0 equivalent to x + offset >= 0.

    Inputs are already quantized to ``input_fraction_bits``. If the comparison
    uses the same precision, no offset is needed. Otherwise the negative
    half-quantum tie rounds to signed zero and therefore remains nonnegative.
    """
    if not 0 <= comparison_fraction_bits <= input_fraction_bits:
        raise ValueError("comparison precision must be between 0 and input precision")
    if comparison_fraction_bits == input_fraction_bits:
        return 0.0
    return 2.0 ** -(comparison_fraction_bits + 1)
