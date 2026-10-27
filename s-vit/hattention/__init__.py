"""hAttention plaintext references and secure-protocol building blocks."""

from .config import HAttentionConfig, HEAD_TYPES, normalize_head_types
from .module import FixedHAttention
from .protocol import (
    FixedPointProfile,
    prepare_grouped_lsa_mixer,
    prepare_lsa,
    secure_grouped_lsa_mixer,
    secure_lsa,
    secure_matmul,
)
from .reference import (
    grouped_hattention,
    legacy_zero_origin_srelu,
    lsa_attention,
    lsa_attention_direct,
    mixer_attention,
    positive_srelu,
    rsa_attention,
)
from .srelu import (
    SReLUMaterial,
    SReLUProfile,
    add_public_half_radius,
    prepare_srelu,
    ruler_srelu,
    ruler_srelu_shifted,
)

__all__ = [
    "FixedHAttention",
    "FixedPointProfile",
    "HAttentionConfig",
    "HEAD_TYPES",
    "grouped_hattention",
    "legacy_zero_origin_srelu",
    "lsa_attention",
    "lsa_attention_direct",
    "mixer_attention",
    "normalize_head_types",
    "positive_srelu",
    "prepare_grouped_lsa_mixer",
    "prepare_lsa",
    "rsa_attention",
    "SReLUMaterial",
    "SReLUProfile",
    "add_public_half_radius",
    "prepare_srelu",
    "ruler_srelu",
    "ruler_srelu_shifted",
    "secure_grouped_lsa_mixer",
    "secure_lsa",
    "secure_matmul",
]
