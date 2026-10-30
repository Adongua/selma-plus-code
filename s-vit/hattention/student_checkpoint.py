"""Load the existing MPCViT student without modifying its source tree.

The legacy module imports a training CLI solely to read linear_gelu. We provide
that one option during import in this process, then restore sys.path/modules.
"""

import argparse
import hashlib
import importlib
from pathlib import Path
import re
import sys
import types

import torch

from .config import HAttentionConfig
from .search import export_fixed_student, make_search_student, search_layers


LEGACY_EXTRA = re.compile(
    r"classifier\.fc_expand\.[01]\.(?:weight|bias)|"
    r"classifier\.blocks\.[0-8]\.linear_expand\.[01]\.(?:weight|bias)|"
    r"classifier\.blocks\.[0-8]\.self_attn\."
    r"(?:E|W|gamma_xnorm|kernel_s|kernel_o|dw_conv\.weight|"
    r"qkv_expand\.[01]\.weight|proj_expand\.[01]\.(?:weight|bias))\Z"
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_mpcvit(mpcvit_root):
    """Import the checked-out model with the explicit legacy linear_gelu=False option."""
    root = Path(mpcvit_root).resolve()
    if not (root / "src" / "vit.py").is_file():
        raise FileNotFoundError(f"MPCViT source not found under {root}")
    existing = sys.modules.get("src")
    if existing is not None:
        src_file = Path(getattr(existing, "__file__", "")).resolve()
        if src_file != root / "src" / "__init__.py":
            raise RuntimeError(f"another src package is already imported: {src_file}")
        transformers = sys.modules.get("src.utils.transformers")
        if transformers is None or getattr(transformers, "lg", None) is not False:
            raise RuntimeError("MPCViT already imported with an incompatible linear_gelu setting")
        return importlib.import_module("src.vit").vit_9_12_64
    sentinel = object()
    original_inference = sys.modules.get("inference", sentinel)
    stub = types.ModuleType("inference")
    stub._parse_args = lambda: (argparse.Namespace(linear_gelu=False), [])
    sys.modules["inference"] = stub
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("src.vit").vit_9_12_64
    finally:
        sys.path.remove(str(root))
        if original_inference is sentinel:
            sys.modules.pop("inference", None)
        else:
            sys.modules["inference"] = original_inference


def load_student(checkpoint_path, mpcvit_root):
    """Load all active model weights; reject missing, incompatible or unknown extras."""
    checkpoint_path = Path(checkpoint_path).resolve()
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("arch") != "vit_9_12_64":
        raise ValueError("expected a vit_9_12_64 MPCViT checkpoint")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("expected a tensor-only state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model = import_mpcvit(mpcvit_root)(pretrained=False)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extras = sorted(set(state) - set(expected))
    unknown = [key for key in extras if not LEGACY_EXTRA.fullmatch(key)]
    if missing or unknown:
        raise ValueError(f"checkpoint mismatch: missing={missing}; unknown extra={unknown}")
    incompatible = [key for key in expected if state[key].shape != expected[key].shape]
    if incompatible:
        raise ValueError(f"checkpoint tensor shape mismatch: {incompatible}")
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    alphas = [block.self_attn.alpha.detach().reshape(-1) for block in model.classifier.blocks]
    if any(alpha.numel() != block.self_attn.num_heads or
           not torch.all((alpha == 0) | (alpha == 1))
           for alpha, block in zip(alphas, model.classifier.blocks)):
        raise ValueError("expected one binary MPCViT alpha per head")
    return model, {"checkpoint": str(checkpoint_path), "sha256": file_sha256(checkpoint_path),
                   "architecture": checkpoint["arch"], "epoch": checkpoint.get("epoch"),
                   "checkpoint_metric_unverified": checkpoint.get("metric"),
                   "active_tensors_loaded": len(expected), "ignored_legacy_keys": extras,
                   "linear_gelu": False, "alpha_convention": "1=RSA, 0=LSA"}


def create_random_student(mpcvit_root):
    """Construct the MPCViT student architecture without loading any weights."""
    model = import_mpcvit(mpcvit_root)(pretrained=False)
    return model, {"checkpoint": None, "sha256": None, "architecture": "vit_9_12_64",
                   "initialization": "random", "linear_gelu": False}


def make_checkpoint_search_student(model, *, prior_strength=1.0, temperature=5.0):
    """Initialize path logits from checkpoint's binary RSA/LSA alpha; keep Mixer open."""
    if not 0 <= prior_strength < float("inf"):
        raise ValueError("prior strength must be finite and nonnegative")
    blocks = model.classifier.blocks
    configs = [HAttentionConfig(block.self_attn.num_heads,
               ("lsa",) * block.self_attn.num_heads, patch_grid=(16, 16)) for block in blocks]
    student = make_search_student(model, configs, remove_attention_dropout=True,
                                  temperature=temperature)
    with torch.no_grad():
        for source, layer in zip(blocks, search_layers(student)):
            alpha = source.self_attn.alpha.reshape(-1).to(layer.structure_logits.device)
            layer.structure_logits.zero_()
            layer.structure_logits[:, 0] = alpha * prior_strength
            layer.structure_logits[:, 1] = (1 - alpha) * prior_strength
    return student


def load_fixed_student(artifact_path, checkpoint_path, mpcvit_root):
    """Restore a search export using the exact original MPCViT initialization."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != "hattention_fixed_student_v1":
        raise ValueError("expected an hAttention fixed-student artifact")
    if artifact.get("student_init", "checkpoint") == "scratch":
        source, metadata = create_random_student(mpcvit_root)
    else:
        if checkpoint_path is None:
            raise ValueError("checkpoint initialization requires its original checkpoint")
        source, metadata = load_student(checkpoint_path, mpcvit_root)
    if artifact.get("source_checkpoint_sha256") != metadata["sha256"]:
        raise ValueError("original MPCViT checkpoint hash does not match export")
    search = make_checkpoint_search_student(source)
    state = artifact.get("state_dict")
    if not isinstance(state, dict) or "search_feature_projection.weight" not in state:
        raise ValueError("fixed-student artifact is missing its feature projection")
    weight = state["search_feature_projection.weight"]
    if weight.ndim != 2 or weight.shape[1] != 192:
        raise ValueError("invalid search feature projection shape")
    search.search_feature_projection = torch.nn.Linear(192, weight.shape[0])
    fixed = export_fixed_student(search, artifact.get("head_types", []))
    fixed.load_state_dict(state, strict=True)
    return fixed, {"artifact": str(Path(artifact_path).resolve()),
                   "sha256": file_sha256(artifact_path), "source": metadata,
                   "head_types": artifact["head_types"]}
