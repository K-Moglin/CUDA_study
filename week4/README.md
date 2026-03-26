# Week4: Simplified FlashAttention-2 (Algorithm 1) CUDA Reproduction

## Overview

This project is a **single-head, FP32, row-wise CUDA reproduction** of the forward pass described by **FlashAttention-2 Algorithm 1**. The codebase focuses on correctness, numerical stability, and a minimal execution path that is easy to study, benchmark, and extend.

At a high level, it computes attention output

\[ O = \mathrm{softmax}(QK^T / \sqrt{d})V ]

with optional **causal masking**, while also producing the per-row log-normalizer:

\[ L_i = m_i + \log l_i ]

where `m` and `l` are the running max and running denominator used by the online softmax formulation.

## Directory / File Structure

```text
week4/
├── main.cu                  # CUDA test driver, benchmarking, CPU reference, correctness checks
├── kernel.cuh               # Core CUDA kernel and block/warp reduction helpers
├── modal_run.py             # Modal cloud runner for H100 compilation and execution
├── flashattn2_alg1_ref.c    # CPU reference implementation of tiled FlashAttention-2 Algorithm 1
├── flashattn.exe            # Prebuilt executable (likely Windows build artifact)
└── __pycache__/             # Python cache directory
```

### What each file does

#### `main.cu`

- Allocates and initializes random `Q`, `K`, `V` matrices.
- Copies data to GPU.
- Launches the CUDA kernel `fa2_alg1_forward_rowwise`.
- Measures average runtime with CUDA events.
- Computes a **CPU naive attention** result for correctness.
- Compares GPU output `O` and log-normalizer `L` against CPU reference values.
- Prints rough throughput in approximate GFLOPs.

#### `kernel.cuh`

Contains the actual CUDA implementation:

- `warpReduceMax` / `warpReduceSum`: warp-level reductions using `__shfl_down_sync`.
- `blockReduceMax` / `blockReduceSum`: block-wide reductions built on top of warp reductions.
- `fa2_alg1_forward_rowwise<Bc, BLOCK_THREADS>`: the main kernel.

This kernel assigns:

- **one CUDA block per query row** `i`
- **one tile of keys/values at a time** along the sequence dimension (`Bc` columns)
- **thread-strided accumulation across feature dimension `d`** for the output vector.

#### `flashattn2_alg1_ref.c`

A CPU implementation of the same tiled online-softmax idea:

- Processes `Q` in row blocks `Br`
- Processes `K/V` in column blocks `Bc`
- Maintains row-wise running `m`, `l`, and `Otilde`

It also includes a naive full-softmax reference for validation. This file is useful for understanding the algorithm before reading the CUDA version.

#### `modal_run.py`

A remote execution script using Modal:

- Builds an image from `nvidia/cuda:12.3.2-devel-ubuntu22.04`
- Uploads `main.cu` and `kernel.cuh`
- Compiles with `nvcc -O3 -std=c++17 -arch=sm_90`
- Runs on an `H100` GPU

This makes the project reproducible even without a local CUDA environment.

#### `flashattn.exe`

A compiled executable included in the folder. It appears to be a build artifact rather than source code and is not required to understand the design.

## Core Algorithm

### 1. Problem being solved

For each row `i`, attention computes:

1. scores: `s_j = <Q_i, K_j> / sqrt(d)`
2. probabilities: `p_j = softmax(s)_j`
3. output: `O_i = sum_j p_j * V_j`

If `causal=1`, entries with `j > i` are masked out.

### 2. Why FlashAttention-style tiling is needed

A naive implementation materializes the full score matrix `QK^T`, which has shape `N x N`. That is expensive in both memory and bandwidth.

This project avoids forming the full score matrix by:

- iterating over `K/V` in tiles of size `Bc`
- maintaining an **online softmax state** per output row
- accumulating the output incrementally

### 3. Online softmax state

For each row, the kernel maintains:

- `m`: running maximum score seen so far
- `l`: running sum of exponentials in the normalized base
- `Otilde`: unnormalized accumulated output vector

When a new tile arrives:

1. compute tile-local max `tile_m`
2. update `m_new = max(m_old, tile_m)`
3. rescale old `l` and `Otilde` by `exp(m_old - m_new)`
4. add current tile’s contribution under the new base
5. finalize with `O = Otilde / l`

This is the key numerical-stability trick that makes FlashAttention work without materializing the full softmax matrix.

## CUDA Execution Model

### Kernel mapping

The main kernel uses:

- `grid.x = N`
- `block.x = BLOCK_THREADS` (default 256)

So each block handles exactly **one query row** `i`.

### Work split inside a block

Inside that block:

- threads `0 .. Bc-1` compute one score each for the current `K/V` tile
- all threads cooperate to reduce the tile max and tile sum
- threads stride over feature dimension `d` to update the row output vector

This is a **row-wise decomposition**.

### Shared memory usage

The kernel stores in shared memory:

- `sh_scores[Bc]`: tile scores
- `sh_probs[Bc]`: tile probabilities after exponentiation

Shared memory is used because all threads in the block need access to the same tile scores/probabilities during the reduction and output accumulation phases.

### Reduction strategy

Instead of a slow global-memory reduction, the implementation uses:

- warp-level shuffle reductions for speed
- shared memory only to combine warp-level partials

This is a common CUDA pattern because it reduces synchronization and shared-memory traffic.

## Important Design Decisions

### 1. **One block per row**

**Decision:** assign one CUDA block to one output row `O[i, :]`.

**Why:**

- matches the row-wise online softmax state naturally (`m`, `l`, `Otilde` are all row-local)
- simplifies causal masking logic (`j > i`)
- makes correctness easier to reason about

**Trade-off:**

- parallelism is exposed mostly across rows, not within the same row beyond tile-level cooperation
- for very small `N`, the GPU may be underutilized

### 2. **Tile only over the sequence dimension (`Bc`)**

**Decision:** split `K/V` columns into tiles of 64, 128, or 256.

**Why:**

- keeps per-tile score/probability data in shared memory
- avoids materializing all `N` scores at once
- mirrors the online-softmax idea from FlashAttention

**Trade-off:**

- this code is simpler than a full 2D tiled implementation, but may leave performance on the table compared with more advanced kernels

### 3. **Store `Otilde` temporarily in output buffer `O`**

**Decision:** reuse the output array `O` as the running unnormalized accumulator.

**Why:**

- avoids allocating a separate GPU buffer for intermediate `Otilde`
- keeps the implementation compact

**Trade-off:**

- mixes “final output” and “temporary state” semantics in the same memory region
- makes the code slightly less explicit for first-time readers

### 4. **Use online max/sum rescaling for numerical stability**

**Decision:** maintain `m` and `l` incrementally and rescale old state when tile max changes.

**Why:**

- prevents overflow/underflow in `exp(score)`
- is the central mathematical idea behind FlashAttention

**Trade-off:**

- adds some control flow and rescaling work per tile
- but this is essential for correctness on real inputs

### 5. **Use warp-shuffle reductions**

**Decision:** implement max/sum reductions with `__shfl_down_sync`.

**Why:**

- avoids slower shared-memory-only reductions
- fits well for block sizes up to 1024 threads

**Trade-off:**

- slightly more CUDA-specific and less beginner-friendly than a plain shared-memory tree reduction

### 6. **CPU naive reference kept in `main.cu`**

**Decision:** validate the CUDA kernel against a plain CPU attention implementation.

**Why:**

- easy to trust and inspect
- provides a correctness baseline independent of the tiled algorithm

**Trade-off:**

- CPU reference becomes slow for large `N`
- suitable for validation, not large-scale benchmarking

### 7. **Separate CPU tiled reference in `flashattn2_alg1_ref.c`**

**Decision:** include a CPU implementation of Algorithm 1 in addition to the naive CPU reference.

**Why:**

- useful for algorithm study
- makes it easier to compare “the same math” on CPU and GPU

**Trade-off:**

- duplicates some logic across files
- but improves readability and pedagogical value

## Execution Flow

### `main.cu`

1. Parse arguments: `N`, `d`, `Bc`, `iters`, `causal`
2. Initialize random host tensors `Q`, `K`, `V`
3. Allocate device memory
4. Copy host data to device
5. Launch kernel once for warmup
6. Launch kernel repeatedly for timing
7. Copy `O` and `L` back to host
8. Compute CPU reference
9. Compare errors and print metrics

### `fa2_alg1_forward_rowwise`

For one row `i`:

1. initialize `m = -inf`, `l = 0`
2. initialize `Otilde = 0`
3. for each `K/V` tile:
   - compute tile scores
   - reduce to tile max
   - update running max/sum base
   - compute tile probabilities
   - reduce tile sum
   - accumulate `p * V` into `Otilde`
4. normalize `Otilde / l`
5. write `L[i] = m + log(l)`

## Build and Run

### Modal run

```bash
python modal_run.py --n 256 --d 64 --bc 128 --iters 50 --causal 1
```

## Known Limitations

This project is intentionally simplified. Compared with production FlashAttention kernels, it currently has several limitations:

1. **Single-head only**

   - no batch dimension
   - no multi-head layout
2. **FP32 only**

   - no half / bf16 / tensor core path
3. **One block per row**

   - simple but not the highest-performance mapping for all regimes
4. **No shared-memory staging of Q/K/V vectors**

   - score computation still loads `Q_i` and `K_j` directly from global memory inside loops
5. **No vectorized loads or Tensor Core MMA**

   - educational rather than peak-performance implementation
6. **No backward pass**

   - forward only
7. **Limited tunable tile sizes**

   - `Bc` only supports `64 / 128 / 256` in the launcher

## Summary

This codebase is best understood as a **teaching-oriented CUDA reproduction** of the FlashAttention-2 Algorithm 1 forward pass:

- it preserves the core online-softmax math
- it supports causal masking
- it validates against a CPU reference
- it is simple enough to study line by line

The most important architectural choice is the **row-wise CUDA mapping with tiled K/V traversal and online softmax rescaling**, which captures the essential FlashAttention idea without introducing the full complexity of an industrial kernel.
