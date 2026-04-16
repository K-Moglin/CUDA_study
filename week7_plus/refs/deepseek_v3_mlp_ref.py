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


def build_mlp_from_weights(
    config: DeepseekV3MoeConfig,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    intermediate_size: int | None = None,
) -> DeepseekV3MLP:
    mlp = DeepseekV3MLP(config, intermediate_size=intermediate_size)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gate_proj_weight)
        mlp.up_proj.weight.copy_(up_proj_weight)
        mlp.down_proj.weight.copy_(down_proj_weight)
    return mlp
