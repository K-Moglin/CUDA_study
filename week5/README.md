# FlashAttention-2 Algorithm 1 CUDA (CuTe Version)

## Overview

This project implements the forward pass of **FlashAttention-2 Algorithm 1** in CUDA using the **CuTe tensor abstraction library**.

Compared to a naive attention implementation, this version:

- Avoids materializing the full attention matrix (O(N²) memory)
- Uses **online softmax** for numerical stability
- Processes keys/values in tiles for better locality

## Project Structure

week5/
├── main.cu          # Host code: testing, benchmarking, CPU reference
├── kernel.cuh       # CUDA kernel (FlashAttention-2 Algorithm 1)
├── modal_run.py     # Run on Modal (H100 GPU)

## Key Design Decisions

### 1. Row-wise Parallelization

- Each CUDA block processes one query row `i`
- Grid size = N
- Simplifies synchronization and data reuse

### 2. Tiling over Key/Value dimension

- Keys/Values are processed in blocks of size `Bc`
- Loop over tiles:

  for t in tiles:
  compute scores
  update softmax
  accumulate output

### 3. Online Softmax (Numerical Stability)

Maintains running `(m, l)`:

m_new = max(m_old, tile_m)
l = l * exp(m_old - m_new) + sum(exp(score - m_new))

Avoids overflow/underflow.

### 4. CuTe Tensor Abstraction

Instead of raw pointer indexing:

Q[i*d + k]

we use:

gQ(i, k)

Benefits:

- Clear semantics
- Explicit layouts
- Easier to extend

### 5. Shared Memory Usage

- `sh_scores`: stores attention scores for a tile
- `sh_probs`: stores softmax probabilities

### 6. Block-level Reduction

Custom reduction for:

- max (for softmax)
- sum (for normalization)

## Performance

Example run:

N=256 d=64 Bc=128 causal=1
time ≈ 0.068 ms
GFLOPs ≈ 247

## Correctness

Compared against CPU naive implementation:

max |O_gpu - O_cpu| ≈ 1e-7
max relative error ≈ 1e-4
max |L_gpu - L_cpu| ≈ 1e-6

Indicating numerically stable and correct results.

## How to Run

### Modal (Cloud GPU)

modal run modal_run.py
