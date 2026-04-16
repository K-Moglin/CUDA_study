from dataclasses import dataclass

import torch


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


class DeepseekV3Route(torch.nn.Module):
    def __init__(self, config: DeepseekV3MoeConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
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


def build_route_from_bias(
    config: DeepseekV3MoeConfig,
    e_score_correction_bias: torch.Tensor | None = None,
) -> DeepseekV3Route:
    route = DeepseekV3Route(config)
    with torch.no_grad():
        if e_score_correction_bias is not None:
            route.e_score_correction_bias.copy_(e_score_correction_bias)
    return route
