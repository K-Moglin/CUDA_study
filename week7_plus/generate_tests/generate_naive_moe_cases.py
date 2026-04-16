import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refs.deepseek_v3_naive_moe_ref import DeepseekV3MoeConfig, build_naive_moe_from_weights
from refs.deepseek_v3_route_ref import build_route_from_bias


OUTPUT_PATH = REPO_ROOT / "test_cases" / "naive_moe_cases.json"


def flatten_tensor(tensor: torch.Tensor) -> list[float] | list[int]:
    return tensor.detach().cpu().reshape(-1).tolist()


def make_weight(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def make_input(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def build_payload() -> dict:
    torch.use_deterministic_algorithms(True)

    config = DeepseekV3MoeConfig()
    batch_size = 2
    seq_len = 3
    num_cases = 3
    seed_weights_base = 423
    seed_inputs_base = 756

    payload = {
        "meta": {
            **asdict(config),
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_cases": num_cases,
            "seed_weights_base": seed_weights_base,
            "seed_inputs_base": seed_inputs_base,
        },
        "cases": [],
    }

    for case_idx in range(num_cases):
        weight_seed = seed_weights_base + case_idx * 17
        input_seed = seed_inputs_base + case_idx * 31
        num_tokens = batch_size * seq_len
        hidden_states = make_input((num_tokens, config.hidden_size), input_seed)
        router_logits = make_input((num_tokens, config.n_routed_experts), input_seed + 1)
        e_score_correction_bias = make_weight((config.n_routed_experts,), weight_seed + 1) * 0.1
        gate_up_proj = make_weight(
            (config.num_local_experts, 2 * config.moe_intermediate_size, config.hidden_size),
            weight_seed + 2,
        )
        down_proj = make_weight(
            (config.num_local_experts, config.hidden_size, config.moe_intermediate_size),
            weight_seed + 3,
        )

        route = build_route_from_bias(config=config, e_score_correction_bias=e_score_correction_bias)
        naive_moe = build_naive_moe_from_weights(config=config, gate_up_proj=gate_up_proj, down_proj=down_proj)
        route.eval()
        naive_moe.eval()

        with torch.no_grad():
            topk_indices, topk_weights = route.route_tokens_to_experts(router_logits)
            routed_out = naive_moe(hidden_states, topk_indices, topk_weights)

        payload["cases"].append(
            {
                "name": f"case_{case_idx}",
                "inputs": {
                    "hidden_states": flatten_tensor(hidden_states),
                    "hidden_states_shape": list(hidden_states.shape),
                    "top_k_index": flatten_tensor(topk_indices.to(torch.int64)),
                    "top_k_index_shape": list(topk_indices.shape),
                    "top_k_weights": flatten_tensor(topk_weights),
                    "top_k_weights_shape": list(topk_weights.shape),
                },
                "weights": {
                    "gate_up_proj": flatten_tensor(gate_up_proj),
                    "gate_up_proj_shape": list(gate_up_proj.shape),
                    "down_proj": flatten_tensor(down_proj),
                    "down_proj_shape": list(down_proj.shape),
                },
                "expected": {
                    "routed_out": flatten_tensor(routed_out),
                    "routed_out_shape": list(routed_out.shape),
                },
                "seeds": {
                    "weights": weight_seed,
                    "inputs": input_seed,
                },
            }
        )

    return payload


def main() -> None:
    first = build_payload()
    second = build_payload()
    first_json = json.dumps(first)
    second_json = json.dumps(second)
    if first_json != second_json:
        raise RuntimeError("NaiveMoe case generation is not deterministic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(first_json + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
