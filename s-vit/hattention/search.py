"""Plaintext differentiable hAttention search primitives.

Search probabilities and the mixed supernet are not part of secure inference.
Costs must be supplied from separately measured, compatible grouped executions.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import HEAD_TYPES, HAttentionConfig
from .reference import lsa_attention, mixer_attention, rsa_attention
from .vit_integration import convert_attention


class SearchableHAttention(nn.Module):
    """Mixed RSA/LSA/Mixer paths for one ViT attention layer."""

    def __init__(self, fixed, *, temperature=5.0, score_bound=16.0,
                 denominator_min=0.01, input_fraction_bits=17):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not math.isfinite(score_bound) or score_bound <= 0 or not math.isfinite(denominator_min) or denominator_min <= 0:
            raise ValueError("numerical bounds must be positive and finite")
        if not isinstance(input_fraction_bits, int) or not 1 <= input_fraction_bits <= 52:
            raise ValueError("invalid input fractional precision")
        self.dim, self.config, self.head_dim = fixed.dim, fixed.config, fixed.head_dim
        self.qkv, self.proj, self.proj_drop = fixed.qkv, fixed.proj, fixed.proj_drop
        self.structure_logits = nn.Parameter(torch.zeros(self.config.num_heads, 3,
                                                        device=self.qkv.weight.device,
                                                        dtype=self.qkv.weight.dtype))
        # Non-persistent so historical checkpoints remain loadable; search mode
        # reconstructs this constraint from the original checkpoint on resume.
        self.register_buffer("allowed_paths", torch.ones(self.config.num_heads, 3, dtype=torch.bool,
                                                          device=self.qkv.weight.device), persistent=False)
        self.fixed_rsa_positions = False
        self.temperature = float(temperature)
        self.path_mode = "soft"
        self.score_bound, self.denominator_min = score_bound, denominator_min
        self.input_fraction_bits = input_fraction_bits
        self.register_buffer("observed_min_denominator", torch.full(
            (self.config.num_heads,), float("inf"), device=self.qkv.weight.device,
            dtype=self.qkv.weight.dtype))
        self.last_numerical_penalty = None

    def probabilities(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        return F.softmax((self.structure_logits / self.temperature).masked_fill(
            ~self.allowed_paths, float("-inf")), dim=-1)

    def forward(self, value):
        if value.ndim != 3 or value.shape[-1] != self.dim:
            raise ValueError(f"input must have (B,N,{self.dim}) layout")
        batch, tokens, _ = value.shape
        q, k, v = self.qkv(value).reshape(batch, tokens, 3, self.config.num_heads,
                                           self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        rho = self.config.resolved_rho(self.head_dim)
        rsa, aux = rsa_attention(q, k, v, epsilon=self.config.epsilon,
                                  radius=self.config.srelu_radius, score_scale=rho,
                                  return_aux=True)
        lsa = lsa_attention(q, k, v, rho)
        mixer = mixer_attention(v, self.config.patch_grid, self.config.has_cls_token)
        probabilities = self.probabilities()
        if self.path_mode == "straight_through":
            hard = F.one_hot(probabilities.argmax(dim=-1), num_classes=3).to(probabilities.dtype)
            probabilities = hard - probabilities.detach() + probabilities
        elif self.path_mode != "soft":
            raise ValueError("invalid path mode")
        paths = torch.stack((rsa, lsa, mixer), dim=-1)
        mixed = (paths * probabilities[None, :, None, None, :]).sum(dim=-1)

        denominator = aux["positive"].sum(dim=-1)
        observer = getattr(self, "denominator_observer", None)
        if observer is not None:
            observer(denominator.detach())
        score_penalty = (aux["scores"].abs() - self.score_bound).clamp_min(0).square().mean(dim=(0, 2, 3))
        denominator_penalty = (self.denominator_min - denominator).clamp_min(0).square().mean(dim=(0, 2))
        unit = 2.0 ** -self.input_fraction_bits
        sensitivity = (unit / denominator.clamp_min(unit)).square().mean(dim=(0, 2))
        self.last_numerical_penalty = ((score_penalty + denominator_penalty + sensitivity)
                                       * probabilities[:, 0]).mean()
        with torch.no_grad():
            observed = denominator.detach().amin(dim=(0, 2))
            self.observed_min_denominator.copy_(torch.minimum(self.observed_min_denominator, observed))
        output = mixed.transpose(1, 2).reshape(batch, tokens, self.dim)
        return self.proj_drop(self.proj(output))


def make_search_student(model, configs, *, remove_attention_dropout=False,
                        temperature=5.0, score_bound=16.0, denominator_min=0.01):
    """Copy a ViTLite model and replace all attention layers with search layers."""
    blocks = getattr(getattr(model, "classifier", None), "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError("model.classifier.blocks must be a ModuleList or Sequential")
    configs = tuple(configs)
    if not blocks or len(configs) != len(blocks):
        raise ValueError("provide one configuration for every block")
    seq_pool = getattr(model.classifier, "seq_pool", None)
    if not isinstance(seq_pool, bool):
        raise TypeError("model.classifier.seq_pool must identify CLS-token usage")
    replacements = []
    for block, config in zip(blocks, configs):
        if not isinstance(config, HAttentionConfig) or config.has_cls_token == seq_pool:
            raise ValueError("invalid attention config or CLS-token configuration")
        fixed = convert_attention(block.self_attn, config,
                                  remove_attention_dropout=remove_attention_dropout)
        replacement = SearchableHAttention(fixed, temperature=temperature,
                                           score_bound=score_bound,
                                           denominator_min=denominator_min)
        replacement.train(block.self_attn.training)
        replacements.append(replacement)
    student = copy.deepcopy(model)
    for block, replacement in zip(student.classifier.blocks, replacements):
        block.self_attn = replacement
    return student


def search_layers(model):
    return [module for module in model.modules() if isinstance(module, SearchableHAttention)]


def grouped_count_distribution(probabilities):
    """Joint probability of (RSA count, Mixer count); path order is HEAD_TYPES."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or probabilities.shape[0] < 1:
        raise ValueError("probabilities must have shape (heads, 3)")
    p = probabilities.new_ones((1, 1))
    for head in probabilities:
        # F.pad preserves autograd and expands one count dimension at a time.
        p = (head[1] * F.pad(p, (0, 1, 0, 1))
             + head[0] * F.pad(p, (0, 1, 1, 0))
             + head[2] * F.pad(p, (1, 0, 0, 1)))
    return p


def expected_group_cost(probabilities, cost_table):
    """Expected cost from a complete [rsa_count,mixer_count] tensor."""
    heads = probabilities.shape[0]
    if tuple(cost_table.shape) != (heads + 1, heads + 1):
        raise ValueError("cost table shape must be (heads+1, heads+1)")
    valid = torch.arange(heads + 1, device=probabilities.device)
    mask = valid[:, None] + valid[None, :] <= heads
    table = cost_table.to(device=probabilities.device, dtype=probabilities.dtype)
    if not torch.isfinite(table[mask]).all() or (table[mask] < 0).any():
        raise ValueError("all feasible group costs must be finite and nonnegative")
    return (grouped_count_distribution(probabilities) * torch.where(mask, table, 0)).sum()


def search_objective(layers, task_loss, cost_tables, budgets, *, lambda_num=0.1,
                     lambda_cost=1.0, lambda_count=0.1, max_rsa=11,
                     lambda_entropy=0.01, entropy_progress=1.0,
                     lambda_task=1.0, lambda_efficiency=0.0,
                     task_loss_scale=1.0):
    """Return structure-stage objective and named components; no implicit costs."""
    if not layers or any(layer.last_numerical_penalty is None for layer in layers):
        raise ValueError("run a forward pass on every search layer first")
    if set(cost_tables) != set(budgets) or not budgets:
        raise ValueError("cost tables and budgets must have the same nonempty metrics")
    if not math.isfinite(task_loss_scale) or task_loss_scale <= 0:
        raise ValueError("task_loss_scale must be positive and finite")
    probs = [layer.probabilities() for layer in layers]
    num = torch.stack([layer.last_numerical_penalty for layer in layers]).mean()
    costs = {}
    for key, budget in budgets.items():
        if not math.isfinite(budget) or budget <= 0 or len(cost_tables[key]) != len(layers):
            raise ValueError("budgets must be positive and each metric needs one table per layer")
        costs[key] = sum(expected_group_cost(p, c) for p, c in zip(probs, cost_tables[key]))
    budget_loss = sum((costs[key] / budget - 1).clamp_min(0).square()
                      for key, budget in budgets.items())
    if not math.isfinite(max_rsa) or max_rsa <= 0:
        raise ValueError("max_rsa must be positive")
    count = (sum(p[:, 0].sum() for p in probs) / max_rsa - 1).clamp_min(0).square()
    entropy = torch.cat([-(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
                         for p in probs]).mean()
    efficiency = sum(costs[key] / budget for key, budget in budgets.items())
    normalized_task = task_loss / task_loss_scale
    loss = (lambda_task * normalized_task + lambda_num * num + lambda_cost * budget_loss
            + lambda_efficiency * efficiency + lambda_count * count
            + lambda_entropy * entropy_progress * entropy)
    return loss, {"task": task_loss, "numerical": num, "budget": budget_loss,
                  "task_normalized": normalized_task, "efficiency": efficiency,
                  "rsa_count": count, "entropy": entropy, "expected_costs": costs}


def weight_parameters(model):
    """Separate trainable weights from architecture logits for alternating updates."""
    logits = {id(layer.structure_logits) for layer in search_layers(model)}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in logits]


def architecture_parameters(model):
    return [layer.structure_logits for layer in search_layers(model)]


def alternating_step(model, weight_optimizer, architecture_optimizer, weight_task,
                     architecture_task, cost_tables, budgets, *, lambda_num=0.1,
                     lambda_cost=1.0, lambda_count=0.1, max_rsa=11,
                     lambda_entropy=0.01, entropy_progress=1.0,
                     lambda_task=1.0, lambda_efficiency=0.0,
                     task_loss_scale=1.0, update_architecture=True):
    """One first-order update on each disjoint batch via caller-provided closures.

    Each closure must run a new forward pass and return a scalar task loss.
    The caller owns data splits, teacher mapping and loss construction.
    """
    layers = search_layers(model)
    weights, architecture = weight_parameters(model), architecture_parameters(model)
    if not weights or not architecture:
        raise ValueError("search model requires trainable weights and architecture parameters")
    for optimizer, params in ((weight_optimizer, weights), (architecture_optimizer, architecture)):
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(params) or {id(p) for p in actual} != {id(p) for p in params}:
            raise ValueError("optimizer parameters do not match their search parameter group")

    def apply(loss, params, optimizer):
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("loss must be a finite scalar")
        optimizer.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for param, grad in zip(params, grads):
            if grad is not None and not torch.isfinite(grad).all():
                raise ValueError("non-finite search gradient")
            param.grad = grad
        optimizer.step()

    for layer in layers:
        layer.last_numerical_penalty = None
    weight_loss = weight_task()
    if any(layer.last_numerical_penalty is None for layer in layers):
        raise ValueError("weight task must run every search layer")
    weight_loss = weight_loss + lambda_num * torch.stack(
        [layer.last_numerical_penalty for layer in layers]).mean()
    apply(weight_loss, weights, weight_optimizer)
    for layer in layers:
        layer.last_numerical_penalty = None
    task = architecture_task()
    arch_loss, components = search_objective(
        layers, task, cost_tables, budgets, lambda_num=lambda_num,
        lambda_cost=lambda_cost, lambda_count=lambda_count, max_rsa=max_rsa,
        lambda_entropy=lambda_entropy, entropy_progress=entropy_progress,
        lambda_task=lambda_task, lambda_efficiency=lambda_efficiency,
        task_loss_scale=task_loss_scale)
    if update_architecture:
        apply(arch_loss, architecture, architecture_optimizer)
    return {"weight_loss": float(weight_loss.detach()),
            "architecture_loss": float(arch_loss.detach()),
            "architecture_components": {k: float(v.detach()) for k, v in components.items()
                                        if k != "expected_costs"},
            "expected_costs": {k: float(v.detach()) for k, v in components["expected_costs"].items()}}


def temperature_at(epoch, epochs, start=5.0, end=0.1):
    if epochs < 1 or epoch < 0 or epoch >= epochs or not 0 < start < float("inf") or not 0 < end < float("inf"):
        raise ValueError("invalid temperature schedule")
    return start * (end / start) ** (epoch / max(epochs - 1, 1))


def discretize(layers, cost_tables, budgets, *, max_steps=10000, cost_preference=0.0,
               require_cls_mixing=None):
    """Choose the highest-probability feasible structure from measured tables.

    RSA is excluded if any observed denominator is below its configured limit
    or the head has never been observed. Observations do not prove future safety.
    Per-layer path-count choices are solved by dynamic programming; a global
    Pareto search handles non-monotone measured group costs across layers.
    """
    if not layers or not budgets or set(cost_tables) != set(budgets):
        raise ValueError("layers, budgets and matching cost tables are required")
    if max_steps < 1 or not math.isfinite(cost_preference) or cost_preference < 0:
        raise ValueError("max_steps must be positive")
    if require_cls_mixing is None:
        require_cls_mixing = any(layer.config.has_cls_token for layer in layers)
    log_probs, allowed = [], []
    for layer in layers:
        p = layer.probabilities().detach().cpu()
        log_probs.append(p.clamp_min(torch.finfo(p.dtype).tiny).log())
        observed = layer.observed_min_denominator.detach().cpu()
        safe_rsa = torch.isfinite(observed) & (observed >= layer.denominator_min)
        # Fixed RSA must remain fixed; retain warnings separately in the report.
        if layer.fixed_rsa_positions:
            safe_rsa = torch.ones_like(safe_rsa)
        allowed.append(layer.allowed_paths.detach().cpu() & torch.stack(
            (safe_rsa, torch.ones_like(observed, dtype=torch.bool),
             torch.ones_like(observed, dtype=torch.bool)), dim=1))
    for key, budget in budgets.items():
        if not math.isfinite(budget) or budget <= 0 or len(cost_tables[key]) != len(layers):
            raise ValueError("invalid budget or number of cost tables")
        for layer, table in zip(layers, cost_tables[key]):
            expected_group_cost(torch.full_like(layer.structure_logits.detach(), 1 / 3), table)
    keys = tuple(budgets)
    layer_options = []
    for layer_index, (scores, masks) in enumerate(zip(log_probs, allowed)):
        # For each feasible (RSA count, Mixer count), retain its most probable
        # head assignment. The dynamic program is O(heads^3), not 3**heads.
        states = {(0, 0): (0.0, ())}
        for head_index in range(scores.shape[0]):
            updated = {}
            for (rsa_count, mixer_count), (score, assignment) in states.items():
                for path in range(3):
                    if not bool(masks[head_index, path]):
                        continue
                    next_rsa = rsa_count + (path == 0)
                    next_mixer = mixer_count + (path == 2)
                    candidate = (score + float(scores[head_index, path]), assignment + (path,))
                    key = (next_rsa, next_mixer)
                    if key not in updated or candidate[0] > updated[key][0]:
                        updated[key] = candidate
            states = updated
        options = []
        for (rsa_count, mixer_count), (score, assignment) in states.items():
            cost = tuple(float(cost_tables[key][layer_index][rsa_count, mixer_count])
                         for key in keys)
            if any(not math.isfinite(value) or value < 0 for value in cost):
                continue
            options.append((cost, score, assignment))
        if not options:
            raise RuntimeError(f"layer {layer_index} has no feasible measured path counts")
        layer_options.append(options)

    # Track whether a head can convey patch information to CLS. A disconnected
    # state must not dominate a connected state during Pareto pruning.
    frontier = [((0.0,) * len(keys), 0.0, (), False)]
    for layer_index, options in enumerate(layer_options):
        candidates = []
        for old_cost, old_score, assignment, connected in frontier:
            for cost, score, paths in options:
                total = tuple(a + b for a, b in zip(old_cost, cost))
                if all(total[i] <= budgets[key] + 1e-12 for i, key in enumerate(keys)):
                    candidates.append((total, old_score + score, assignment + (paths,),
                                       connected or any(path != 2 for path in paths)))
        if not candidates:
            minimum = {key: min(option[0][i] for option in layer_options[:layer_index + 1])
                       for i, key in enumerate(keys)}
            raise RuntimeError(f"no structure meets all budgets after layer {layer_index}; "
                               f"per-layer minimum costs are {minimum}")
        # Pareto-prune: a state is useful only if no state is both cheaper in
        # every metric and at least as likely under the learned path logits.
        candidates.sort(key=lambda row: (sum(row[0]), row[0], -row[1]))
        kept = []
        for candidate in candidates:
            cost, score, _, connected = candidate
            dominated = any(other[3] == connected and all(other[0][i] <= cost[i] + 1e-12 for i in range(len(keys)))
                            and other[1] >= score - 1e-12 for other in kept)
            if dominated:
                continue
            kept = [other for other in kept
                    if not (other[3] == connected and all(cost[i] <= other[0][i] + 1e-12 for i in range(len(keys)))
                            and score >= other[1] - 1e-12)]
            kept.append(candidate)
        frontier = kept
    total_heads = sum(layer.config.num_heads for layer in layers)
    if require_cls_mixing:
        frontier = [state for state in frontier if state[3]]
        if not frontier:
            raise RuntimeError("no budget-feasible structure connects CLS to patch tokens")
    costs, score, selections, _ = max(frontier, key=lambda row: row[1] / total_heads - cost_preference *
                                   sum(row[0][i] / budgets[key] for i, key in enumerate(keys)))
    return {"head_types": [[HEAD_TYPES[i] for i in heads] for heads in selections],
            "costs": dict(zip(keys, costs)), "budgets": dict(budgets),
            "log_probability_score": score,
            "cost_preference": cost_preference,
            "requires_cls_mixing": require_cls_mixing,
            "discretization": "global measured-cost Pareto dynamic program",
            "rsa_observation_warning": "observed samples only; not a safety proof"}


def export_fixed_student(model, head_types, *, allow_disconnected_cls=False):
    """Export a separate fixed-path model, preserving learned projections."""
    layers = search_layers(model)
    head_types = tuple(head_types)
    if not layers or len(head_types) != len(layers):
        raise ValueError("one head path list is required per search layer")
    if (not allow_disconnected_cls and getattr(model.classifier, "seq_pool", None) is False
            and all(layer.config.has_cls_token for layer in layers)
            and all(path == "mixer" for paths in head_types for path in paths)):
        raise ValueError("all-Mixer CLS classifier cannot receive image patch information")
    configs = [HAttentionConfig(num_heads=layer.config.num_heads, head_types=paths,
                               rho=layer.config.rho, epsilon=layer.config.epsilon,
                               srelu_radius=layer.config.srelu_radius,
                               has_cls_token=layer.config.has_cls_token,
                               patch_grid=layer.config.patch_grid)
               for layer, paths in zip(layers, head_types)]
    # The numerical penalty is a non-leaf tensor from the last forward pass;
    # omit its ephemeral autograd graph while copying the model.
    memo = {id(layer.last_numerical_penalty): None for layer in layers
            if layer.last_numerical_penalty is not None}
    fixed = copy.deepcopy(model, memo)
    from .module import FixedHAttention
    index = 0
    for block in fixed.classifier.blocks:
        layer = block.self_attn
        if isinstance(layer, SearchableHAttention):
            replacement = FixedHAttention(layer.dim, configs[index],
                                           qkv_bias=layer.qkv.bias is not None,
                                           projection_bias=layer.proj.bias is not None)
            replacement.qkv, replacement.proj = layer.qkv, layer.proj
            replacement.proj_drop = layer.proj_drop
            replacement.train(layer.training)
            block.self_attn = replacement
            index += 1
    if index != len(layers):
        raise ValueError("search layers must reside in classifier.blocks[*].self_attn")
    return fixed
