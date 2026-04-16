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


def build_router_from_weights(
    config: DeepseekV3MoeConfig,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor | None = None,
) -> DeepseekV3TopkRouter:
    router = DeepseekV3TopkRouter(config)
    with torch.no_grad():
        router.weight.copy_(router_weight)
        if e_score_correction_bias is not None:
            router.e_score_correction_bias.copy_(e_score_correction_bias)
    return router
