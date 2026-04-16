from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DeepseekV3MoeConfig:
    hidden_size: int = 8
    intermediate_size: int = 16
    moe_intermediate_size: int = 4
    n_routed_experts: int = 4
    num_local_experts: int = 4
    n_shared_experts: int = 1
    n_group: int = 2
    topk_group: int = 1
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    hidden_act: str = "silu"


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


ACT2FN = {
    "silu": silu,
}


class DeepseekV3MLP(torch.nn.Module):
    def __init__(self, config: DeepseekV3MoeConfig, intermediate_size: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3TopkRouter(torch.nn.Module):
    def __init__(self, config: DeepseekV3MoeConfig):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.weight = torch.nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        return F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))


class DeepseekV3NaiveMoe(torch.nn.Module):
    def __init__(self, config: DeepseekV3MoeConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = torch.nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV3MoE(torch.nn.Module):
    def __init__(self, config: DeepseekV3MoeConfig):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        routed_out = self.experts(flat_hidden_states, topk_indices, topk_weights).view(*orig_shape)
        shared_out = self.shared_experts(residuals)
        final_out = routed_out + shared_out
        return router_logits, topk_indices, topk_weights, routed_out, shared_out, final_out


def build_moe_from_weights(
    config: DeepseekV3MoeConfig,
    router_weight: torch.Tensor,
    routed_gate_up_proj: torch.Tensor,
    routed_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None = None,
) -> DeepseekV3MoE:
    moe = DeepseekV3MoE(config)
    with torch.no_grad():
        moe.gate.weight.copy_(router_weight)
        if e_score_correction_bias is not None:
            moe.gate.e_score_correction_bias.copy_(e_score_correction_bias)
        moe.experts.gate_up_proj.copy_(routed_gate_up_proj)
        moe.experts.down_proj.copy_(routed_down_proj)
        moe.shared_experts.gate_proj.weight.copy_(shared_gate_proj)
        moe.shared_experts.up_proj.weight.copy_(shared_up_proj)
        moe.shared_experts.down_proj.weight.copy_(shared_down_proj)
    return moe
