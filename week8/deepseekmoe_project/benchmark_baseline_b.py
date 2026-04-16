#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Minimal Transformers-style DeepSeekV3 MoE baseline
# Adapted to be standalone and benchmark-friendly.
# -----------------------------

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


ACT2FN = {
    "gelu": gelu,
    "silu": silu,
}


@dataclass
class MiniDeepseekV3Config:
    hidden_size: int = 512
    moe_intermediate_size: int = 1024
    intermediate_size: int = 1024
    hidden_act: str = "silu"
    n_routed_experts: int = 8
    num_local_experts: int = 8
    n_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    num_experts_per_tok: int = 2
    initializer_range: float = 0.02


class DeepseekV3MLP(nn.Module):
    def __init__(self, config: MiniDeepseekV3Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config: MiniDeepseekV3Config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        return F.linear(hidden_states.float(), self.weight.float())


class DeepseekV3NaiveMoe(nn.Module):
    """Standalone version of the naive expert collection used by the Transformers DeepSeekV3 MoE."""

    def __init__(self, config: MiniDeepseekV3Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)

        for expert_idx_t in expert_hit:
            expert_idx = int(expert_idx_t[0].item())
            if expert_idx >= self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DeepseekV3MoE(nn.Module):
    def __init__(self, config: MiniDeepseekV3Config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits: torch.Tensor):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        experts_per_group = self.n_routed_experts // self.n_group
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, experts_per_group)
            .topk(min(2, experts_per_group), dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=min(self.topk_group, self.n_group), dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def init_model_weights(model: nn.Module, std: float = 0.02):
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=std)
        else:
            torch.nn.init.zeros_(p)


def make_model(args) -> DeepseekV3MoE:
    config = MiniDeepseekV3Config(
        hidden_size=args.hidden,
        moe_intermediate_size=args.moe_intermediate,
        intermediate_size=args.moe_intermediate,
        hidden_act=args.act,
        n_routed_experts=args.experts,
        num_local_experts=args.experts,
        n_shared_experts=args.shared,
        n_group=args.n_group,
        topk_group=args.topk_group,
        norm_topk_prob=not args.disable_norm_topk,
        routed_scaling_factor=args.routed_scaling_factor,
        num_experts_per_tok=args.topk,
        initializer_range=args.init_std,
    )
    model = DeepseekV3MoE(config)
    init_model_weights(model, std=args.init_std)
    return model


def benchmark_torch(model: nn.Module, x: torch.Tensor, warmup: int, iters: int):
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / iters
    out = model(x)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return avg_ms, out


def benchmark_torch_compiled(model: nn.Module, x: torch.Tensor, warmup: int, iters: int):
    compiled_model = torch.compile(model, mode="max-autotune")
    return benchmark_torch(compiled_model, x, warmup, iters)


def parse_cuda_binary_ms(stdout: str) -> Optional[float]:
    m = re.search(r"Average forward time:\s*([0-9.]+)\s*ms", stdout)
    if m:
        return float(m.group(1))
    return None


def build_default_cuda_args(args):
    tokens_local = args.tokens_local
    if tokens_local is None:
        if args.nranks <= 0:
            raise ValueError("--nranks must be > 0")
        total_tokens = args.batch * args.seq
        if total_tokens % args.nranks != 0:
            raise ValueError(
                f"total_tokens={total_tokens} is not divisible by nranks={args.nranks}; "
                "please pass --tokens-local explicitly"
            )
        tokens_local = total_tokens // args.nranks

    return [
        "--T_local", str(tokens_local),
        "--H", str(args.hidden),
        "--I", str(args.cuda_intermediate if args.cuda_intermediate is not None else args.moe_intermediate),
        "--E", str(args.experts),
        "--shared", str(args.shared),
        "--topk", str(args.topk),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
    ]


def parse_cuda_args(cuda_args):
    if cuda_args is None:
        return []
    if isinstance(cuda_args, list):
        return cuda_args
    return shlex.split(cuda_args)


def run_cuda_binary(args) -> tuple[Optional[float], Optional[int], Optional[list[str]]]:
    if not args.cuda_binary:
        return None, None, None

    if args.cuda_args:
        extra_args = parse_cuda_args(args.cuda_args)
    else:
        extra_args = build_default_cuda_args(args)

    cmd = [args.cuda_binary] + extra_args
    env = os.environ.copy()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    print("\n===== CUDA binary command =====")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("\n===== CUDA binary output =====")
    print(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"CUDA binary failed with exit code {proc.returncode}")

    avg_ms = parse_cuda_binary_ms(proc.stdout)
    if avg_ms is None:
        raise RuntimeError("Could not parse 'Average forward time: ... ms' from CUDA binary output")
    return avg_ms, (args.tokens_local if args.tokens_local is not None else (args.batch * args.seq) // args.nranks), cmd


def main():
    parser = argparse.ArgumentParser(description="Baseline B: Transformers-style DeepSeekV3 MoE benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--nranks", type=int, default=2, help="Used to infer tokens-local when not provided")
    parser.add_argument("--tokens-local", type=int, default=None, help="Per-rank tokens for the custom CUDA binary; defaults to batch*seq/nranks")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--moe-intermediate", type=int, default=2048)
    parser.add_argument("--cuda-intermediate", type=int, default=None, help="Optional separate intermediate for your CUDA binary")
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--shared", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--n-group", type=int, default=1)
    parser.add_argument("--topk-group", type=int, default=1)
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--disable-norm-topk", action="store_true")
    parser.add_argument("--act", type=str, default="silu", choices=["silu", "gelu"])
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile", action="store_true", help="Benchmark torch.compile(model) too")
    parser.add_argument("--cuda-binary", type=str, default=None, help="Path to your custom CUDA MoE executable")
    parser.add_argument("--cuda-args", nargs=argparse.REMAINDER, default=None, help="Raw args forwarded to the CUDA binary. Example: --cuda-args --T_local 512 --H 1024 --I 2048 --E 8 --shared 1 --topk 2 --warmup 10 --iters 50")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    model = make_model(args).to(device=device)
    if device.type == "cuda":
        model = model.to(dtype=dtype)

    x = torch.randn(
        args.batch,
        args.seq,
        args.hidden,
        device=device,
        dtype=dtype if device.type == "cuda" else torch.float32,
    )

    total_tokens = args.batch * args.seq
    inferred_tokens_local = args.tokens_local if args.tokens_local is not None else total_tokens // args.nranks

    print("===== Baseline B configuration =====")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"batch={args.batch} seq={args.seq} total_tokens={total_tokens}")
    print(f"hidden={args.hidden} moe_intermediate={args.moe_intermediate} experts={args.experts} topk={args.topk} shared={args.shared}")
    print(f"n_group={args.n_group} topk_group={args.topk_group} act={args.act}")
    print(f"nranks={args.nranks} inferred_tokens_local={inferred_tokens_local}")

    torch_ms, out = benchmark_torch(model, x, args.warmup, args.iters)
    torch_toks = total_tokens / (torch_ms / 1000.0)

    compiled_ms = None
    compiled_toks = None
    if args.compile:
        compiled_ms, _ = benchmark_torch_compiled(model, x, args.warmup, args.iters)
        compiled_toks = total_tokens / (compiled_ms / 1000.0)

    cuda_ms, cuda_tokens_local, _ = run_cuda_binary(args)
    cuda_toks = None if cuda_ms is None else (cuda_tokens_local * args.nranks) / (cuda_ms / 1000.0)

    print("\n===== Benchmark summary =====")
    print(f"Transformers-style baseline avg forward: {torch_ms:.3f} ms")
    print(f"Transformers-style throughput: {torch_toks:.2f} tokens/s")
    if compiled_ms is not None:
        print(f"Transformers-style + torch.compile avg forward: {compiled_ms:.3f} ms")
        print(f"Transformers-style + torch.compile throughput: {compiled_toks:.2f} tokens/s")
    if cuda_ms is not None:
        print(f"Custom CUDA avg forward: {cuda_ms:.3f} ms")
        print(f"Custom CUDA throughput (assuming {args.nranks} ranks x tokens-local): {cuda_toks:.2f} tokens/s")
        print(f"Speedup (CUDA / baseline): {torch_ms / cuda_ms:.3f}x")
        if compiled_ms is not None:
            print(f"Speedup (CUDA / compiled baseline): {compiled_ms / cuda_ms:.3f}x")

    print("\n===== Sample output stats =====")
    print(f"output shape={tuple(out.shape)}")
    print(f"output mean={out.float().mean().item():.6f}")
    print(f"output std={out.float().std().item():.6f}")


if __name__ == "__main__":
    main()
