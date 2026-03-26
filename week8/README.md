# DeepSeekMoE CUDA + NCCL (Modal, single-process multi-GPU)

This project implements a correctness-first distributed DeepSeekMoE forward operator with:
- CUDA kernels for linear/GELU/gather/scatter/combine
- NCCL `ncclAlltoAll` for dispatch and return
- data parallelism over local tokens
- expert parallelism over routed experts
- shared experts replicated on each GPU
- CPU reference checking for correctness

## Files
- `deepseek_moe_modal.cu`: main implementation
- `run_modal.py`: Modal entrypoint
- `reference_moe.cpp`: lightweight CPU reference snippet
- `README.md`: usage notes

## Run on Modal
```bash
modal run run_modal.py
```

With custom parameters:
```bash
modal run run_modal.py --hidden 16 --intermediate 32 --tokens 8 --experts 4 --topk 2 --shared 1 --iters 20
```

## Pipeline
1. Local shared-expert branch
2. Local router logits
3. Host-side top-k routing (week7-style ordering)
4. Pack token-expert pairs by destination GPU
5. First NCCL all-to-all dispatch
6. Local expert FFN on owner GPU
7. Second NCCL all-to-all return
8. Unpermute + weighted combine + add shared output

## Notes
- This is a baseline implementation focused on clarity and correctness.
- Routing/permutation metadata is handled on the host side to keep the implementation simple.
- Dispatch uses fixed-capacity padding so it fits `ncclAlltoAll`.
