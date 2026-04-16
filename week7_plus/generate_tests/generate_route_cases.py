import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refs.deepseek_v3_route_ref import DeepseekV3MoeConfig, build_route_from_bias


OUTPUT_PATH = REPO_ROOT / "test_cases" / "route_cases.json"


def flatten_tensor(tensor: torch.Tensor) -> list[float] | list[int]:
    return tensor.detach().cpu().reshape(-1).tolist()


def make_input(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def make_bias(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1


def build_payload() -> dict:
    torch.use_deterministic_algorithms(True)

    config = DeepseekV3MoeConfig()
    batch_size = 2
    seq_len = 3
    num_cases = 3
    seed_weights_base = 323
    seed_inputs_base = 656

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
        router_logits = make_input((num_tokens, config.n_routed_experts), input_seed)
        e_score_correction_bias = make_bias((config.n_routed_experts,), weight_seed + 1)

        route = build_route_from_bias(
            config=config,
            e_score_correction_bias=e_score_correction_bias,
        )
        route.eval()

        with torch.no_grad():
            topk_indices, topk_weights = route.route_tokens_to_experts(router_logits)

        payload["cases"].append(
            {
                "name": f"case_{case_idx}",
                "inputs": {
                    "router_logits": flatten_tensor(router_logits),
                    "router_logits_shape": list(router_logits.shape),
                },
                "weights": {
                    "e_score_correction_bias": flatten_tensor(e_score_correction_bias),
                    "e_score_correction_bias_shape": list(e_score_correction_bias.shape),
                },
                "expected": {
                    "topk_indices": flatten_tensor(topk_indices.to(torch.int64)),
                    "topk_indices_shape": list(topk_indices.shape),
                    "topk_weights": flatten_tensor(topk_weights),
                    "topk_weights_shape": list(topk_weights.shape),
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
        raise RuntimeError("Route case generation is not deterministic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(first_json + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
