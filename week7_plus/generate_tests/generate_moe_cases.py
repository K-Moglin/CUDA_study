import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refs.deepseek_v3_moe_ref import DeepseekV3MoeConfig, build_moe_from_weights


OUTPUT_PATH = REPO_ROOT / "test_cases" / "moe_cases.json"


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
    seed_weights_base = 523
    seed_inputs_base = 856

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

    shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts

    for case_idx in range(num_cases):
        weight_seed = seed_weights_base + case_idx * 17
        input_seed = seed_inputs_base + case_idx * 31
        hidden_states = make_input((batch_size, seq_len, config.hidden_size), input_seed)
        router_weight = make_weight((config.n_routed_experts, config.hidden_size), weight_seed + 1)
        e_score_correction_bias = make_weight((config.n_routed_experts,), weight_seed + 2) * 0.1
        routed_gate_up_proj = make_weight(
            (config.num_local_experts, 2 * config.moe_intermediate_size, config.hidden_size),
            weight_seed + 3,
        )
        routed_down_proj = make_weight(
            (config.num_local_experts, config.hidden_size, config.moe_intermediate_size),
            weight_seed + 4,
        )
        shared_gate_proj = make_weight((shared_intermediate_size, config.hidden_size), weight_seed + 5)
        shared_up_proj = make_weight((shared_intermediate_size, config.hidden_size), weight_seed + 6)
        shared_down_proj = make_weight((config.hidden_size, shared_intermediate_size), weight_seed + 7)

        moe = build_moe_from_weights(
            config=config,
            router_weight=router_weight,
            routed_gate_up_proj=routed_gate_up_proj,
            routed_down_proj=routed_down_proj,
            shared_gate_proj=shared_gate_proj,
            shared_up_proj=shared_up_proj,
            shared_down_proj=shared_down_proj,
            e_score_correction_bias=e_score_correction_bias,
        )
        moe.eval()

        with torch.no_grad():
            router_logits, topk_indices, topk_weights, routed_out, shared_out, final_out = moe(hidden_states)

        payload["cases"].append(
            {
                "name": f"case_{case_idx}",
                "inputs": {
                    "hidden_states": flatten_tensor(hidden_states),
                    "hidden_states_shape": list(hidden_states.shape),
                },
                "weights": {
                    "router_weight": flatten_tensor(router_weight),
                    "router_weight_shape": list(router_weight.shape),
                    "e_score_correction_bias": flatten_tensor(e_score_correction_bias),
                    "e_score_correction_bias_shape": list(e_score_correction_bias.shape),
                    "routed_gate_up_proj": flatten_tensor(routed_gate_up_proj),
                    "routed_gate_up_proj_shape": list(routed_gate_up_proj.shape),
                    "routed_down_proj": flatten_tensor(routed_down_proj),
                    "routed_down_proj_shape": list(routed_down_proj.shape),
                    "shared_gate_proj": flatten_tensor(shared_gate_proj),
                    "shared_gate_proj_shape": list(shared_gate_proj.shape),
                    "shared_up_proj": flatten_tensor(shared_up_proj),
                    "shared_up_proj_shape": list(shared_up_proj.shape),
                    "shared_down_proj": flatten_tensor(shared_down_proj),
                    "shared_down_proj_shape": list(shared_down_proj.shape),
                },
                "expected": {
                    "router_logits": flatten_tensor(router_logits),
                    "router_logits_shape": list(router_logits.shape),
                    "topk_indices": flatten_tensor(topk_indices.to(torch.int64)),
                    "topk_indices_shape": list(topk_indices.shape),
                    "topk_weights": flatten_tensor(topk_weights),
                    "topk_weights_shape": list(topk_weights.shape),
                    "routed_out": flatten_tensor(routed_out),
                    "routed_out_shape": list(routed_out.shape),
                    "shared_out": flatten_tensor(shared_out),
                    "shared_out_shape": list(shared_out.shape),
                    "final_out": flatten_tensor(final_out),
                    "final_out_shape": list(final_out.shape),
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
        raise RuntimeError("MoE case generation is not deterministic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(first_json + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
