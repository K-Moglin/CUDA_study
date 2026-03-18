# Week 7 Assignment – DeepSeekV3 MoE Operator (Pure C Implementation)

## 1. Overview

This project implements the **Mixture-of-Experts (MoE) operator** used in DeepSeekV3, following the design described in the *DeepSeekMoE* paper.

The goal of this assignment is to:

- Understand the architecture of MoE layers
- Generate deterministic test cases using a reference implementation
- Implement the MoE operator in **pure C (no parallelism, no CUDA)**
- Verify correctness using block-level and end-to-end tests

---

## 2. Background

In a standard Transformer, each layer contains a single Feed-Forward Network (FFN).
In MoE, this FFN is replaced by multiple **experts**, and a **router (gate)** selects a subset of experts for each token.

The output of a standard MoE layer is:

output = Σ (g_i * Expert_i(x))

where:

- g_i = routing weight
- Expert_i = FFN of expert i

---

## 3. DeepSeekMoE Design

DeepSeekMoE extends standard MoE with:

### 3.1 Routed Experts

- Selected by the router (top-k)
- Each token activates only a small subset

### 3.2 Shared Experts

- Always active for all tokens
- Capture general/common knowledge

### 3.3 Final Output

final_output = routed_output + shared_output

This project implements both routed experts and shared experts.

---

## 4. Project Structure

week7_moe/
│
├── mini_deepseek_moe.py     # Python reference implementation
├── generate_tests.py        # Generate deterministic test cases
├── test_case_shared.npz     # Saved test data
├── moe.c                   # Pure C implementation
├── test_runner.py           # Validation script
├── moe.dll / moe.so         # Compiled shared library
└── README.md

---

## 5. Reference Implementation (Python)

A minimal MoE model is implemented using NumPy:

### Components:

- router_forward → compute expert probabilities (softmax)
- topk_routing → select top-k experts
- expert_forward → FFN (GELU activation)
- moe_forward_with_shared → full MoE pipeline

### Key Features:

- Fixed random seed for reproducibility
- Small dimensions for easier debugging
- Block-level intermediate outputs

---

## 6. Test Case Generation

Test cases are generated using:

python generate_tests.py

The following data is saved:

Input:

- x
- gate_w
- experts_w1, experts_w2
- shared_w1, shared_w2

Intermediate outputs:

- probs
- topk_idx
- topk_weights
- expert_outputs
- shared_output
- routed_output

Final output:

- final_output

---

## 7. C Implementation

The MoE operator is implemented in pure C:

### Implemented Functions:

- softmax_inplace
- gelu_scalar
- router_forward
- topk_routing
- expert_forward
- moe_forward_shared

### Key Design Choices:

- No parallelism (single-threaded)
- Static dimensions for simplicity
- Memory handled using raw pointers
- Explicit loops for matrix operations

---

## 8. Validation

Run:

python test_runner.py

Validation includes:

1. Router output (probs)
2. Top-k indices
3. Top-k weights
4. Shared expert output
5. Routed output
6. Final MoE output

Example Output:

Router max err: ~1e-12
Shared max err: ~1e-6
Routed max err: ~1e-6
Final max err : ~1e-6

ALL SHARED-MOE TESTS PASS

---

## 9. Results

- All block-level tests pass
- Numerical error is within acceptable floating-point tolerance (< 1e-5)
- C implementation matches Python reference

---

## 10. Key Insights

### 10.1 Sparse Computation

Only top-k experts are activated → reduces computation cost

### 10.2 Expert Specialization

Different experts learn different patterns

### 10.3 Shared + Routed Experts

- Shared experts → general knowledge
- Routed experts → specialized knowledge


---

## 11. Conclusion

This project successfully implements and validates a DeepSeek-style MoE operator in pure C.

The implementation demonstrates:

- Correct routing behavior
- Accurate expert computation
- Consistent numerical results with reference model

This provides a solid foundation for further work on high-performance MoE systems and multi-GPU training.
