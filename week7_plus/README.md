# DeepSeekV3 MoE C Reproduction

## Overview

This project extracts the DeepSeekV3 MoE blocks from Hugging Face `modeling_deepseek_v3.py`, generates deterministic fake-weight test cases, and re-implements the same semantics in plain C with block-level and end-to-end correctness checks.

## Scope

Target reference blocks:

- `DeepseekV3MLP`
- `DeepseekV3TopkRouter`
- `DeepseekV3MoE.route_tokens_to_experts`
- `DeepseekV3NaiveMoe`
- `DeepseekV3MoE.forward`

Reference of truth:

- `transformers/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`

## DeepSeekMoE Design Notes

This repo focuses on the two design ideas called out by DeepSeekMoE and used by DeepSeekV3:

- Fine-grained expert segmentation: routed capacity is partitioned into many smaller experts and a token selects multiple experts.
- Shared expert isolation: a shared MLP path always runs alongside routed experts and is added to the routed output.

For implementation details, the Hugging Face `modeling_deepseek_v3.py` file is treated as the source of truth whenever it differs from the paper wording.

## Planned Layout

The workspace root acts as the requested `project/` root.

- `transformers/`: local Hugging Face checkout
- `refs/`: minimal Python reference blocks
- `generate_tests/`: deterministic JSON case generators
- `test_cases/`: generated self-contained cases
- `c_impl/`: plain C implementation and test runner
- `scripts/`: helper scripts such as config inspection

## Current HF Semantics

The current Hugging Face implementation uses:

- `DeepseekV3MLP`: gated MLP, `down_proj(act(gate_proj(x)) * up_proj(x))`
- `DeepseekV3TopkRouter`: float32 linear projection to router logits
- `route_tokens_to_experts`: `sigmoid`, optional group masking via top-2-per-group score sums, then token-wise top-k selection, optional top-k normalization, then `routed_scaling_factor`
- `DeepseekV3NaiveMoe`: per-expert packed `gate_up_proj` plus `down_proj`, scatter-gather accumulation by selected experts
- `DeepseekV3MoE.forward`: `router -> route -> routed experts -> shared experts -> add`

## HF Block Responsibilities

- `DeepseekV3MLP`: shared-expert gated MLP, `down_proj(act(gate_proj(x)) * up_proj(x))`
- `DeepseekV3TopkRouter`: float32 linear projection from token hidden states to expert logits
- `DeepseekV3MoE.route_tokens_to_experts`: `sigmoid -> group score computation -> top group mask -> top-k expert choice -> optional normalization -> routed scaling`
- `DeepseekV3NaiveMoe`: per-expert gated MLP over routed experts, weighted accumulation into token outputs
- `DeepseekV3MoE.forward`: end-to-end MoE path, `router -> route -> routed experts -> shared experts -> final add`

## File List

- `refs/deepseek_v3_mlp_ref.py`
- `refs/deepseek_v3_topk_router_ref.py`
- `refs/deepseek_v3_route_ref.py`
- `refs/deepseek_v3_naive_moe_ref.py`
- `refs/deepseek_v3_moe_ref.py`
- `generate_tests/generate_mlp_cases.py`
- `generate_tests/generate_router_cases.py`
- `generate_tests/generate_route_cases.py`
- `generate_tests/generate_naive_moe_cases.py`
- `generate_tests/generate_moe_cases.py`
- `test_cases/mlp_cases.json`
- `test_cases/router_cases.json`
- `test_cases/route_cases.json`
- `test_cases/naive_moe_cases.json`
- `test_cases/moe_cases.json`
- `c_impl/moe.h`
- `c_impl/moe.c`
- `c_impl/test_moe.c`
- `scripts/print_deepseek_v3_moe_config.py`
- `requirements.txt`
- `Makefile`

## Environment

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Print relevant HF config fields:

```bash
python scripts/print_deepseek_v3_moe_config.py
```

## Generate Test Cases

Generate everything:

```bash
make gen-all
```

On Windows in this workspace, `mingw32-make` is available even when `make` is not on `PATH`.

Generate a single block:

```bash
python generate_tests/generate_mlp_cases.py
python generate_tests/generate_router_cases.py
python generate_tests/generate_route_cases.py
python generate_tests/generate_naive_moe_cases.py
python generate_tests/generate_moe_cases.py
```

Each generator:

- uses fake weights and fake inputs only
- separates weight and input seeds
- enables deterministic PyTorch execution
- generates JSON with config, seeds, inputs, weights, and expected outputs
- runs the same generation flow twice and asserts byte-for-byte equality

## Build And Test

Build the C test runner:

```bash
make build
```

Run all correctness checks:

```bash
make test-all
```

The test runner:

- reads the JSON files from `test_cases/`
- reconstructs tensors in row-major layout
- compares float tensors with absolute/relative tolerance
- compares indices exactly
- prints the first mismatch when a block fails

## Execution Order

Implementation proceeds in this order, with generator + C implementation + tests at each step:

1. `DeepseekV3MLP`
2. `DeepseekV3TopkRouter`
3. `DeepseekV3MoE.route_tokens_to_experts`
4. `DeepseekV3NaiveMoe`
5. `DeepseekV3MoE.forward`

## Determinism Rules

- Fake weights only
- Fake inputs only
- Separate seeds for weights and inputs
- Deterministic generation checks must pass twice in a row
- JSON cases must include config, weights, inputs, expected outputs, and intermediate tensors where needed

## Known Assumptions And Limitations

- The workspace root is used as the requested `project/` root so the existing local `transformers/` checkout can be reused directly.
- Hugging Face `modeling_deepseek_v3.py` is treated as the implementation source of truth, even where the resulting behavior is more subtle than a high-level paper summary.
- No real DeepSeek weights are used anywhere in this repo.
- The C implementation is intentionally scalar and CPU-only: no CUDA, no threading, no BLAS.
- JSON parsing in `c_impl/test_moe.c` is minimal and tailored to the generated test file structure.
- The current config is intentionally tiny for deterministic correctness testing rather than performance benchmarking.
