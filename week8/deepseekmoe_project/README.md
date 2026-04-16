# Multi-GPU DeepSeekMoE CUDA + NCCL Project

This project is a submission-ready baseline for the assignment:

> Implement multi-gpu DeepseekMoE operator in CUDA using NCCL using data-parallelism and expert-parallelism.
> - Check that the generated test cases from the assignment pass.
> - Compare the performance of your implementation with Transformers for large datasets.

## What is included

- `deepseek_moe_nccl_alltoall.cu`
  - Two-GPU CUDA implementation of a simplified DeepSeek-style MoE forward pass.
  - Uses **true NCCL all-to-all communication pattern** via grouped `ncclSend`/`ncclRecv`.
  - Uses data parallelism over local tokens and expert parallelism over routed experts.
  - Includes built-in **CPU reference checking** and prints `TEST PASS` / `TEST FAIL`.

- `generate_assignment_tests.py`
  - Generates a JSON file containing deterministic test configurations.

- `check_assignment_tests.py`
  - Builds the CUDA binary and runs all generated test cases.
  - Fails if any case does not print `TEST PASS`.

- `benchmark_baseline_b.py`
  - Transformers-style DeepSeekV3 MoE baseline (performance comparison baseline B).
  - Can benchmark eager / `torch.compile` and compare against the CUDA binary.

- `run_modal.py`
  - Modal launcher for 2x H100.

## Semantics of the CUDA implementation

The CUDA implementation follows the week7/week8 simplified MoE semantics:

1. Router logits: `x @ gate_w`
2. Softmax over all experts
3. Top-k expert selection
4. Renormalize top-k routing probabilities
5. Per-expert FFN: `GELU(x @ w1) @ w2`
6. Sum weighted routed expert outputs
7. Add shared expert output

This is the correctness target used by the CPU reference and assignment test checker.

## Build

```bash
nvcc -O3 -std=c++17 -o deepseek_moe_nccl_alltoall deepseek_moe_nccl_alltoall.cu -lnccl -lm
```

## Run correctness check

```bash
./deepseek_moe_nccl_alltoall --check --T_local 4 --H 8 --I 16 --E 4 --shared 1 --topk 2 --iters 5 --warmup 1
```

You should see output similar to:

```text
max |gpu-cpu| = ...
max rel error = ...
TEST PASS
```

## Generate and run assignment tests

```bash
python generate_assignment_tests.py --output assignment_test_cases.json
python check_assignment_tests.py --binary ./deepseek_moe_nccl_alltoall --cases assignment_test_cases.json
```

## Run performance comparison with baseline B

Only baseline B:

```bash
python benchmark_baseline_b.py --device cuda --dtype float16 --batch 8 --seq 128 --hidden 1024 --moe-intermediate 2048 --experts 8 --topk 2 --shared 1 --warmup 10 --iters 50
```

Compare baseline B with your CUDA binary:

```bash
python benchmark_baseline_b.py --device cuda --dtype float16 --batch 8 --seq 128 --hidden 1024 --moe-intermediate 2048 --experts 8 --topk 2 --shared 1 --warmup 10 --iters 50 --cuda-binary ./deepseek_moe_nccl_alltoall
```

## Notes

- The CUDA implementation uses **true NCCL all-to-all pattern** implemented with grouped `ncclSend`/`ncclRecv`.
- The performance baseline is **Transformers-style DeepSeekV3 MoE routing**, intended for benchmarking rather than correctness checking.
- The built-in CPU reference is the correctness oracle for the simplified CUDA assignment target.
