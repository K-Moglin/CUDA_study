import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refs.deepseek_v3_topk_router_ref import DeepseekV3MoeConfig, build_router_from_weights


OUTPUT_PATH = REPO_ROOT / "test_cases" / "router_cases.json"


def flatten_tensor(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


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
    seed_weights_base = 223
    seed_inputs_base = 556

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
        hidden_states = make_input((batch_size, seq_len, config.hidden_size), input_seed)
        router_weight = make_weight((config.n_routed_experts, config.hidden_size), weight_seed + 1)
        e_score_correction_bias = torch.zeros(config.n_routed_experts, dtype=torch.float32)

        router = build_router_from_weights(
            config=config,
            router_weight=router_weight,
            e_score_correction_bias=e_score_correction_bias,
        )
        router.eval()

        with torch.no_grad():
            router_logits = router(hidden_states)

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
                },
                "expected": {
                    "router_logits": flatten_tensor(router_logits),
                    "router_logits_shape": list(router_logits.shape),
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
        raise RuntimeError("Router case generation is not deterministic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(first_json + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
