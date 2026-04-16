# Week 9 Report: Reimplementing DeepSeekMoE with ThunderKittens on B200

## 1. Project goal

This project rewrites the main DeepSeekMoE execution path so that the expensive expert MLP GEMMs can use a ThunderKittens-backed path on **NVIDIA B200**, while preserving the distributed NCCL all-to-all MoE structure of the original implementation.

The final goal is not just to run a standalone ThunderKittens example, but to connect ThunderKittens into the **MoE main path itself** and measure whether that improves the end-to-end model execution.

---

## 2. Reference architecture and implementation scope

The official `modeling_deepseek_v3.py` shows that the DeepSeekV3 MoE block is composed of:
- a router that produces top-k expert assignments,
- routed experts,
- shared experts,
- and a final sum of routed output plus shared-expert output. fileciteturn13file18turn13file15

The official file also uses a gated MLP form for the expert implementation, i.e. `down_proj(act(gate_proj(x)) * up_proj(x))`. fileciteturn13file17

My CUDA project keeps the same **high-level MoE execution pattern**:
- route tokens,
- dispatch tokens to experts,
- run per-expert MLP compute,
- all-to-all return,
- combine routed outputs,
- add shared expert outputs.

For this assignment, the optimization target is the **expert MLP compute path** and the surrounding dispatch/combine overhead, rather than reproducing every detail of the full official Python implementation.

---

## 3. Baseline implementation

The starting point was a custom distributed MoE forward implementation based on:
- NCCL all-to-all for token dispatch and return,
- a custom router,
- per-rank local experts,
- shared experts,
- and a baseline CUDA linear kernel for expert GEMMs.

The early baseline version was correct, but slow. In previous benchmark runs, the baseline achieved only `0.071x`, `0.083x`, and `0.119x` relative speedup against the PyTorch reference benchmark, depending on the revision. fileciteturn13file5turn13file6turn13file4

That baseline already passed the generated correctness tests with relative error on the order of `1e-7`. fileciteturn13file5turn13file6turn13file4

---

## 4. How ThunderKittens was integrated into the MoE path

The final CUDA file introduces a ThunderKittens-backed execution path directly inside the MoE main code, instead of only compiling the standalone TK GEMM demo.

### 4.1 Compile-time gate

At the top of the CUDA source, a compile-time switch controls whether the ThunderKittens path is enabled:

```cpp
#ifndef USE_TK_GEMM
#define USE_TK_GEMM 0
#endif

#if USE_TK_GEMM
#include "kittens.cuh"
#include "prototype.cuh"
#endif
```

This keeps a safe fallback path for small correctness tests while allowing the TK path to be enabled for the B200 build.

### 4.2 Centralized GEMM dispatch

The project centralizes expert GEMM execution through a dedicated entry instead of spreading GEMM logic across many places. The key function is:

- `linear_forward_tk_ws(...)`

This function is the ThunderKittens-backed linear layer path. It is called from:

- `expert_forward_gpu_ws(...)`

which executes the two expert MLP GEMMs:
- first projection: `x @ w1`
- second projection: `hidden @ w2`

By routing both shared-expert and routed-expert GEMMs through this one entry point, the code makes the ThunderKittens path the actual compute backend for MoE expert MLPs.

### 4.3 BF16 weights for TK path

ThunderKittens on B200 is most useful when feeding tensor cores with low-precision operands. To support that, each expert stores both:
- baseline `float` weights,
- and resident `__nv_bfloat16` copies (`d_w1_bf16`, `d_w2_bf16`).

That allows the TK path to avoid per-call weight conversion.

### 4.4 Reusable workspace instead of repeated allocations

A large early overhead came from repeated `cudaMalloc/cudaFree` in hot paths. The optimized version moves long-lived buffers into `RankBuffers`, including:
- shared-expert temporary buffers,
- routed-expert input/output scratch,
- BF16 staging buffers (`d_tk_a`, `d_tk_c`),
- device-side top-k metadata (`d_topk_idx`, `d_topk_w`),
- a small device counter (`d_count`).

This is important because a ThunderKittens path only helps if the surrounding execution is not dominated by allocator overhead.

### 4.5 Device-side dispatch packing

Another major source of overhead was host participation in MoE routing. The optimized code moves more of that work onto the GPU.

Two important kernels are:
- `pack_dispatch_kernel(...)`
- `build_rows_scales_kernel(...)`

These reduce MoE main-path overhead by:
- packing dispatch payloads on device,
- building local expert row lists on device,
- reducing unnecessary host round-trips before expert compute.

That change matters because expert GEMM acceleration alone is not enough if dispatch and repacking remain the true bottleneck.

---

## 5. Build configuration for ThunderKittens on B200

The standalone ThunderKittens B200 GEMM example was first used to verify that the environment and compiler flags were correct. The successful build used the ThunderKittens-provided Makefile and B200-specific flags such as:
- `--expt-extended-lambda`
- `--expt-relaxed-constexpr`
- `-DKITTENS_BLACKWELL`
- `-gencode arch=compute_100a,code=sm_100a` fileciteturn13file6turn13file11

Once that toolchain was confirmed to work, the same project was extended so that the **MoE main executable** also had a TK-enabled build target.

---

## 6. Experimental progression

The project evolved through several stages:

### Stage A: baseline only
- correctness passed,
- but performance was poor (`0.071x` to `0.119x`). fileciteturn13file5turn13file6turn13file4

### Stage B: TK environment verified
- the B200 standalone GEMM demo built and ran successfully,
- with multiple templates reaching from roughly `368 TFLOPs` up to about `1708 TFLOPs`, confirming the ThunderKittens environment was functional on B200. fileciteturn13file9turn13file8

### Stage C: first TK-connected MoE main path
- correctness still passed,
- but performance was initially worse than baseline (`0.023x` in one revision), showing that naïve TK integration is not enough. fileciteturn10file0

### Stage D: reduced non-GEMM overhead
- reusable BF16 weights,
- reusable workspace,
- reduced host participation,
- device-side dispatch packing,
- device-side row/scales construction.

After these changes, the project reached the final benchmark below.

---

## 7. Final results

In the final validated run:
- the TK main build succeeded,
- correctness tests `tiny`, `small`, and `medium` all passed,
- relative error stayed at about `1e-7`,
- and the TK-enabled MoE main path became faster than the baseline. fileciteturn10file0

### Final benchmark summary

- Baseline version: `0.494x`
- TK main version: `1.697x` fileciteturn10file0

This implies an improvement of approximately:

```text
1.697 / 0.494 ≈ 3.43x
```

So the final ThunderKittens-enabled MoE main path is about **3.4x faster** than the project’s baseline implementation under the measured benchmark setting. fileciteturn10file0

The output statistics also remained identical between baseline and TK main benchmark output:
- `output shape=(8, 128, 1024)`
- `output mean=-0.000361`
- `output std=0.252831` fileciteturn10file0

This is useful evidence that the speedup did not come from producing an inconsistent result.

---

## 8. Why performance improved

The final speedup did not come from one change alone. It came from the combination of:

1. **Using ThunderKittens for expert GEMM execution**
   - pushing expert matrix multiplications toward a B200-appropriate tensor-core path.

2. **Keeping BF16 expert weights resident**
   - avoiding repeated per-call weight conversion.

3. **Reusing scratch buffers**
   - reducing allocator overhead inside the MoE hot path.

4. **Reducing host-side routing/repacking overhead**
   - moving dispatch packing and local row construction onto the device.

5. **Preserving fallback behavior for small shapes**
   - correctness-oriented small cases do not force the project into an inefficient TK path where tensor-core tiling is not a good fit.

In practice, that combination is what allowed the TK path to move from “correct but slower” to “correct and faster.”

---

## 9. How to run the project

### Files
- `deepseek_moe_nccl_alltoall.cu`: main CUDA implementation with baseline and TK-backed paths
- `run_modal.py`: Modal runner for build, correctness tests, and benchmark
- `ThunderKittens/`: vendored ThunderKittens source tree

### Run on Modal

```bash
modal run run_modal.py
```

### What the Modal script does
- builds the baseline executable,
- builds the TK-enabled main executable,
- generates assignment tests,
- runs correctness checks,
- runs benchmark on baseline,
- runs benchmark on TK main.

### Typical benchmark setting used in the final run
- `batch=8`
- `seq=128`
- `nranks=2`
- `hidden=1024`
- `moe_intermediate=2048`
- `experts=8`
- `topk=2`
- `shared=1` fileciteturn10file0

---

## 10. Limitations and future work

Although the final version already outperforms the project baseline, it is still not the end of the optimization story.

Possible next steps include:
- grouped expert execution so that more experts can share larger GEMM work units,
- further reduction of dispatch/combine overhead,
- better overlap of NCCL communication with expert compute,
- direct tensor-memory-oriented kernels for a wider set of shapes,
- extending the implementation from the simplified expert MLP path toward more of the full official DeepSeekV3 expert formulation.

---

## 11. Conclusion

This project moved from a correct but slow distributed DeepSeekMoE baseline to a ThunderKittens-enabled B200 implementation that actually accelerates the MoE main path. The final version preserves correctness on all generated tests and improves the measured benchmark performance from `0.494x` to `1.697x`, which is about a **3.4x speedup over the project baseline**. fileciteturn10file0

The key lesson is that using ThunderKittens effectively in MoE is not only about replacing GEMM calls. It also requires reducing the surrounding routing, packing, allocation, and data-conversion overhead so that tensor-core acceleration can dominate the actual runtime.
