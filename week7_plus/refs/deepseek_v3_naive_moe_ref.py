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


def build_naive_moe_from_weights(
    config: DeepseekV3MoeConfig,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> DeepseekV3NaiveMoe:
    moe = DeepseekV3NaiveMoe(config)
    with torch.no_grad():
        moe.gate_up_proj.copy_(gate_up_proj)
        moe.down_proj.copy_(down_proj)
    return moe
