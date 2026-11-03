"""Experimental three-party full ViTLite forward with a public test fixture.

Ring3 material is generated just before use; provider operations by CrypTen
remain online. Original MPCViT is explicitly adapted to sReLU RSA, sGeLU and
sLayerNorm. This is a numerical experiment, not a validated privacy protocol.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sgelu.runtime_cuda import CudaRing3, CudaRing3Wide
from sgelu.protocol import prepare_g2, ruler_g2
from sgelu.wide_range import WideGeLUProfile, prepare_wide_gelu, wide_gelu
from slayernorm.protocol import Profile as LNProfile, prepare as prepare_ln, layernorm
from slayernorm.wide_range import WideLayerNormProfile, prepare_wide_layernorm, wide_layernorm
from slayernorm.benchmark_crypten_cuda import _aggregate, _barrier, _crypten_measure, _plain
from .config import HAttentionConfig
from .hybrid import prepare_hybrid, secure_hybrid
from .hybrid_wide_range import prepare_wide_hybrid, secure_wide_hybrid
from .reference import rsa_attention
from .student_checkpoint import file_sha256, load_fixed_student, load_student
from .vit_executor import forward_vit
from .vit_integration import make_fixed_student


def numeric_profile_details(name, mu):
    if name == "wide-range":
        ln = WideLayerNormProfile()
        return {"layernorm_input_bound": 64, "layernorm_affine_bound": 4,
                "layernorm_variance_fraction_bits": 28, "layernorm_scale": 16,
                "layernorm_epsilon": ln.epsilon, "layernorm_effective_epsilon": ln.effective_epsilon,
                "gelu_input_bound": 12, "gelu_mu": mu, "qkv_input_bound": 16,
                "rsa_score_bound_exclusive": 16, "rsa_denominator_bounds": [0.0005, 2048],
                "rsa_reciprocal_fraction_bits": 24, "rsa_reciprocal_iterations": 32}
    if name == "legacy":
        return {"layernorm_input_bound": 4, "layernorm_affine_bound": 2,
                "gelu_input_bound": 8, "gelu_mu": mu, "qkv_input_bound": 8,
                "rsa_score_bound_exclusive": 16, "rsa_denominator_bounds": [0.01, 64],
                "rsa_reciprocal_fraction_bits": 24, "rsa_reciprocal_iterations": 24}
    raise ValueError("unknown numerical profile")


def load_model(checkpoint, root, fixed):
    if fixed:
        model, meta = load_fixed_student(fixed, checkpoint, root)
        return model.eval(), meta
    source, meta = load_student(checkpoint, root)
    paths = [tuple("rsa" if int(x) else "lsa" for x in b.self_attn.alpha.detach().reshape(-1))
             for b in source.classifier.blocks]
    configs = [HAttentionConfig(len(p), p, patch_grid=(16, 16)) for p in paths]
    model = make_fixed_student(source, configs, remove_attention_dropout=True).eval()
    return model, {"source": meta, "head_types": paths, "artifact": None}


def image_tensor(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BICUBIC),
                            dtype=np.float32).copy()
    x = torch.from_numpy(pixels).permute(2, 0, 1) / 255
    mean = x.new_tensor((0.480, 0.448, 0.397)).reshape(3, 1, 1)
    std = x.new_tensor((0.272, 0.265, 0.274)).reshape(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def audit_public_fixture(model, image, numeric_profile="legacy"):
    """Inspect plaintext ranges on this public image, never secret intermediates."""
    if numeric_profile not in ("legacy", "wide-range"):
        raise ValueError("unknown numerical profile")
    ln_bound, affine_bound = (64, 4) if numeric_profile == "wide-range" else (4, 2)
    gelu_bound = 12 if numeric_profile == "wide-range" else 8
    qkv_bound = 16 if numeric_profile == "wide-range" else 8
    denominator_min, denominator_max = ((0.0005, 2048) if numeric_profile == "wide-range" else (0.01, 64))
    observations, hooks = [], []
    def norm_hook(name):
        def inspect(module, inputs):
            value = inputs[0]
            maximum = float(value.abs().max())
            gamma = float(module.weight.detach().abs().max())
            beta = float(module.bias.detach().abs().max())
            observations.append({"site": name, "type": "layernorm", "input_max_abs": maximum,
                                 "gamma_max_abs": gamma, "beta_max_abs": beta,
                                 "within_profile": maximum <= ln_bound and gamma <= affine_bound and beta <= affine_bound})
        return inspect
    def gelu_hook(name):
        def inspect(_module, _inputs, output):
            maximum = float(output.abs().max())
            observations.append({"site": name, "type": "gelu", "input_max_abs": maximum,
                                 "within_profile": maximum <= gelu_bound})
        return inspect
    def attention_hook(name):
        def inspect(module, inputs):
            value = inputs[0]
            b, n, dim = value.shape
            h = module.config.num_heads
            q, k, v = module.qkv(value).reshape(b, n, 3, h, dim // h).permute(2, 0, 3, 1, 4).unbind(0)
            indexes = [i for i, path in enumerate(module.config.head_types) if path == "rsa"]
            maximum = max(float(t.abs().max()) for t in (q, k, v))
            row = {"site": name, "type": "attention", "qkv_max_abs": maximum,
                   "within_profile": maximum <= qkv_bound}
            if indexes:
                idx = torch.tensor(indexes, device=q.device)
                _, aux = rsa_attention(q.index_select(1, idx), k.index_select(1, idx),
                    v.index_select(1, idx), epsilon=round(1e-5 * 2**17) / 2**17, return_aux=True)
                score = float(aux["scores"].abs().max())
                low, high = float(aux["denominator"].min()), float(aux["denominator"].max())
                row.update(score_max_abs=score, denominator_min=low, denominator_max=high)
                row["within_profile"] &= score < 16 and low >= denominator_min and high <= denominator_max
            observations.append(row)
        return inspect
    try:
        for index, block in enumerate(model.classifier.blocks):
            prefix = f"block_{index + 1}"
            hooks.append(block.pre_norm.register_forward_pre_hook(norm_hook(prefix + ".pre_norm")))
            hooks.append(block.norm1.register_forward_pre_hook(norm_hook(prefix + ".norm1")))
            hooks.append(block.linear1.register_forward_hook(gelu_hook(prefix + ".gelu")))
            hooks.append(block.self_attn.register_forward_pre_hook(attention_hook(prefix + ".attention")))
        hooks.append(model.classifier.norm.register_forward_pre_hook(norm_hook("classifier.norm")))
        with torch.no_grad():
            logits, _ = model(image)
        return logits.double(), observations
    finally:
        for hook in hooks:
            hook.remove()


class SecureOperators:
    def __init__(self, base, wide, mu, numeric_profile="legacy"):
        from crypten.cuda import CUDALongTensor
        from crypten.mpc import MPCTensor
        self.base, self.wide, self.mu = base, wide, mu
        self.numeric_profile = numeric_profile
        self.cuda_tensor, self.mpc_tensor = CUDALongTensor, MPCTensor
        self.offline_seconds = 0.0
        self.offline_bytes = 0
        self.offline_phases = 0

    def prepare(self, fn):
        torch.cuda.synchronize(self.base.device)
        start, sent, phases = time.perf_counter(), self.base.sent_bytes, self.base.phases
        result = fn()
        torch.cuda.synchronize(self.base.device)
        self.offline_seconds += time.perf_counter() - start
        self.offline_bytes += self.base.sent_bytes - sent
        self.offline_phases += self.base.phases - phases
        return result

    def wrap(self, share):
        return self.mpc_tensor.from_shares(self.cuda_tensor(share), precision=17)

    def conv(self, x, layer):
        out = x.conv2d(layer.weight.detach(), stride=layer.stride, padding=layer.padding)
        return out + layer.bias.detach().view(1, -1, 1, 1) if layer.bias is not None else out

    def cat_cls(self, x, cls):
        import crypten
        # x-x is a public zero in every share; adding the public CLS introduces it once.
        zero = x[:, :1, :] - x[:, :1, :]
        return crypten.cat([zero + cls.detach(), x], dim=1)

    def linear(self, x, layer):
        out = x.matmul(layer.weight.detach().t())
        return out + layer.bias.detach() if layer.bias is not None else out

    def norm(self, x, layer):
        wide_range = getattr(self, "numeric_profile", "legacy") == "wide-range"
        profile = WideLayerNormProfile(epsilon=layer.eps) if wide_range else LNProfile(epsilon=layer.eps)
        prepare_fn = prepare_wide_layernorm if wide_range else prepare_ln
        function = wide_layernorm if wide_range else layernorm
        material = self.prepare(lambda: prepare_fn(self.base, self.wide, tuple(x.shape), profile))
        gamma = self.base.public(torch.round(layer.weight.detach() * (1 << 17)).long())
        beta = self.base.public(torch.round(layer.bias.detach() * (1 << 17)).long())
        return self.wrap(function(self.base, self.wide, x.share.data, gamma, beta, material))

    def gelu(self, x):
        if getattr(self, "numeric_profile", "legacy") == "wide-range":
            profile = WideGeLUProfile(mu=self.mu)
            material = self.prepare(lambda: prepare_wide_gelu(self.base, self.wide, tuple(x.shape), profile))
            return self.wrap(wide_gelu(self.base, self.wide, x.share.data, material))
        material = self.prepare(lambda: prepare_g2(self.base, self.wide, tuple(x.shape), self.mu))
        return self.wrap(ruler_g2(self.base, self.wide, x.share.data, material))

    def attention(self, q, k, v, module):
        config = module.config
        if (config.srelu_radius != 0.0055 or config.epsilon != 1e-5 or
                config.resolved_rho(q.shape[-1]) != q.shape[-1] ** -0.5):
            raise ValueError("unsupported attention profile for Ring3 RSA")
        wide_range = getattr(self, "numeric_profile", "legacy") == "wide-range"
        prepare_fn = prepare_wide_hybrid if wide_range else prepare_hybrid
        function = secure_wide_hybrid if wide_range else secure_hybrid
        material = self.prepare(lambda: prepare_fn(self.base, self.wide, tuple(q.shape),
                                config.head_types, rho=config.resolved_rho(q.shape[-1])))
        result = function(self.base, self.wide, q.share.data, k.share.data, v.share.data,
                          material, patch_grid=config.patch_grid, has_cls_token=config.has_cls_token)
        return self.wrap(result)


def worker(rank, rendezvous, options):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ.update(RANK=str(rank), WORLD_SIZE="3", RENDEZVOUS=rendezvous,
                      DISTRIBUTED_BACKEND="gloo")
    import crypten
    crypten.cfg.encoder.precision_bits = 17
    crypten.init(device=device)
    crypten.cfg.communicator.verbose = True
    try:
        model, meta = load_model(options["checkpoint"], options["mpcvit_root"], options["fixed_student"])
        model = model.to(device).eval()
        image = image_tensor(options["image"][0]).to(device)
        reference, observations = audit_public_fixture(model, image, options["numeric_profile"])
        violations = [row for row in observations if not row["within_profile"]]
        if rank == 0:
            print(json.dumps({"event": "public_fixture_range_audit", "violations": violations}), flush=True)
        if violations and not options["allow_out_of_range"]:
            raise ValueError("public fixture exceeds protocol bounds; see range audit; do not use result as a valid benchmark")
        base = CudaRing3(device=device, reveal_mode="p2p")
        wide = CudaRing3Wide(base)
        records = []
        for iteration in range(options["repeats"] + 1):
            shared, sharing = _crypten_measure(device, crypten, lambda: crypten.cryptensor(
                image, src=0, precision=17, device=device))
            torch.cuda.reset_peak_memory_stats(device)
            ops = SecureOperators(base, wide, options["mu"], options["numeric_profile"])
            _barrier(device)
            crypten.reset_communication_stats()
            torch.cuda.synchronize(device)
            sent, phases, start = base.sent_bytes, base.phases, time.perf_counter()
            def progress(index, _value):
                if rank == 0:
                    print(json.dumps({"event": "layer_complete", "warmup": iteration == 0,
                                      "layer": index + 1}), flush=True)
            with torch.no_grad():
                output = forward_vit(model, shared, ops, on_layer=progress)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - start
            stats = crypten.communicator.get().get_communication_stats()
            crypto = _aggregate(device, 0, stats, bidirectional_counter=True)
            ring = _aggregate(device, wall, {"bytes": base.sent_bytes - sent,
                "rounds": base.phases - phases, "time": 0.0})
            offline = _aggregate(device, ops.offline_seconds, {"bytes": ops.offline_bytes,
                "rounds": ops.offline_phases, "time": 0.0})
            online = _aggregate(device, max(0.0, wall - ops.offline_seconds), {
                "bytes": base.sent_bytes - sent - ops.offline_bytes,
                "rounds": base.phases - phases - ops.offline_phases, "time": 0.0})
            opened, reconstruction = _crypten_measure(device, crypten, lambda: output.get_plain_text())
            actual = _plain(opened)
            diff = (actual - reference).abs()
            finite = bool(torch.isfinite(actual).all())
            peak = torch.tensor([torch.cuda.max_memory_allocated(device)], dtype=torch.int64)
            peaks = [torch.zeros_like(peak) for _ in range(3)]
            dist.all_gather(peaks, peak)
            record = {"warmup": iteration == 0, "input_sharing": sharing,
                      "forward_wall": ring, "ring_preprocessing": offline,
                      "online_ring_excluding_preprocessing": online,
                      "online_crypten": crypto, "output_reconstruction": reconstruction,
                      "peak_allocated_bytes_per_rank": [int(p.item()) for p in peaks],
                      "error_vs_plaintext_logits": {"finite": finite,
                          "mae": float(diff.mean()) if finite else None,
                          "max_abs": float(diff.max()) if finite else None,
                          "plaintext_top1": int(reference.argmax(-1).item()),
                          "secure_top1": int(actual.argmax(-1).item()) if finite else None}}
            if rank == 0:
                records.append(record)
                print(json.dumps({"event": "iteration_complete", "record": record}), flush=True)
        if rank == 0:
            paths = [list(b.self_attn.config.head_types) for b in model.classifier.blocks]
            measured = [r for r in records if not r["warmup"]]
            passed = not violations and all(r["error_vs_plaintext_logits"]["finite"] and
                         r["error_vs_plaintext_logits"]["max_abs"] <= options["max_abs_error"]
                         for r in measured)
            report = {"status": "benchmark_passed_not_security_validated" if passed else "benchmark_failed_correctness", "backend": "gloo", "world_size": 3,
                "scope": "full ViTLite forward; public model weights and public benchmark image secret-shared for execution; batch 1",
                "model": meta, "head_types": paths,
                "path_counts": dict(Counter(p for layer in paths for p in layer)),
                "image_sha256": file_sha256(options["image"][0]), "repeats": options["repeats"],
                "public_fixture_range_audit": observations,
                "precision": "Q17", "numeric_profile": options["numeric_profile"],
                "numeric_profile_details": numeric_profile_details(options["numeric_profile"], options["mu"]),
                "mu": options["mu"], "max_abs_error_threshold": options["max_abs_error"], "records": records,
                "crypten_import": crypten.__file__, "python": platform.python_version(), "torch": torch.__version__,
                "timing_note": "forward wall includes just-in-time preprocessing; online time subtracts local preprocessing before rank maximum; CrypTen provider work remains online; sync instrumentation included",
                "communication_note": "Ring3 sent payloads and CrypTen bidirectional logs/2 are separate; counters are not equivalent protocol rounds",
                "numerics_note": "sLayerNorm and sGeLU approximate the trained LayerNorm and GeLU; RSA uses sReLU and bounded reciprocal; secret ranges are unchecked",
                "security_note": "Ruler openings and CUDA PRG are not privacy validated; baseline is sReLU-adapted MPCViT, not the original ReLU protocol"}
            out = Path(options["output"])
            out.mkdir(parents=True, exist_ok=False)
            with (out / "comparison.json").open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write("\n")
            print(json.dumps({"report": str(out / "comparison.json"), "status": report["status"]}), flush=True)
    finally:
        crypten.uninit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mpcvit-root", required=True)
    parser.add_argument("--fixed-student")
    parser.add_argument("--image", required=True, nargs="+", help="public benchmark image(s); only audit-only supports multiple")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mu", type=float, default=1.4621998268680836)
    parser.add_argument("--max-abs-error", type=float, default=1.0,
                        help="exploratory max logit error gate against the plaintext model")
    parser.add_argument("--allow-out-of-range", action="store_true",
                        help="diagnostic only; range failures mark the result invalid")
    parser.add_argument("--audit-only", action="store_true",
                        help="inspect plaintext numerical ranges on CPU without initializing three-party runtime")
    parser.add_argument("--numeric-profile", choices=("legacy", "wide-range"), default="legacy",
                        help="select legacy operators or the experimental wide-range implementations")
    args = parser.parse_args()
    if args.repeats < 1 or not 0.25 <= args.mu <= 4 or not 0 < args.max_abs_error < float("inf"):
        parser.error("invalid repeats, mu or max-abs-error")
    if Path(args.output).exists():
        parser.error("output already exists")
    if args.audit_only:
        model, meta = load_model(args.checkpoint, args.mpcvit_root, args.fixed_student)
        rows = []
        for image in args.image:
            logits, observations = audit_public_fixture(model, image_tensor(image), args.numeric_profile)
            violations = [row for row in observations if not row["within_profile"]]
            rows.append({"image": str(Path(image).resolve()), "sha256": file_sha256(image),
                         "top1": int(logits.argmax(-1).item()), "violations": violations,
                         "observations": observations})
            print(json.dumps({"event": "audit_complete", "image": image,
                              "violations": len(violations)}), flush=True)
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False)
        report = {"status": "plaintext_range_audit_only", "numeric_profile": args.numeric_profile,
                  "numeric_profile_details": numeric_profile_details(args.numeric_profile, args.mu), "model": meta, "images": rows,
                  "note": "Public fixture and plaintext intermediate values only. Passing these observations does not verify secret fixed-point ranges or security."}
        with (root / "range-audit.json").open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({"report": str(root / "range-audit.json"),
                          "images": len(rows), "total_violations": sum(len(x["violations"]) for x in rows)}))
        return
    if len(args.image) != 1:
        parser.error("secure run requires exactly one image")
    if torch.cuda.device_count() < 3:
        parser.error("three CUDA devices required")
    with tempfile.TemporaryDirectory(prefix="hattention-vit-") as tmp:
        mp.spawn(worker, args=(Path(tmp, "store").as_uri(), vars(args)), nprocs=3, join=True)


if __name__ == "__main__":
    main()
