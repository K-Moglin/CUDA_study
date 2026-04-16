from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
if str(TRANSFORMERS_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


@dataclass
class MoeConfigSnapshot:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    n_shared_experts: int
    n_routed_experts: int
    num_local_experts: int
    n_group: int | None
    topk_group: int | None
    num_experts_per_tok: int | None
    norm_topk_prob: bool | None
    routed_scaling_factor: float
    hidden_act: str


def main() -> None:
    config = DeepseekV3Config()
    snapshot = MoeConfigSnapshot(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        n_shared_experts=config.n_shared_experts,
        n_routed_experts=config.n_routed_experts,
        num_local_experts=config.num_local_experts,
        n_group=config.n_group,
        topk_group=config.topk_group,
        num_experts_per_tok=config.num_experts_per_tok,
        norm_topk_prob=config.norm_topk_prob,
        routed_scaling_factor=config.routed_scaling_factor,
        hidden_act=config.hidden_act,
    )
    for key, value in asdict(snapshot).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
