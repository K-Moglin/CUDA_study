"""Benchmark nano-sglang throughput.

Compare scheduler-based continuous batching against generating one request
at a time with the plain engine.
"""

from __future__ import annotations

import argparse
import time

from nano_sglang.engine import Engine
from nano_sglang.sampling import SamplingParams
from nano_sglang.scheduler import Scheduler

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "Write two concise sentences about why batching helps LLM inference."


def build_prompts(base_prompt: str, num_requests: int) -> list[str]:
    """Create prompts with small suffix differences to avoid identical requests."""
    return [f"{base_prompt} Request {i}." for i in range(num_requests)]


def count_generated_tokens(engine: Engine, texts: list[str]) -> int:
    """Count generated tokens using the same tokenizer as inference."""
    return sum(len(engine.tokenizer.encode(text)) for text in texts)


def run_serial(
    model_path: str,
    prompts: list[str],
    sampling_params: SamplingParams,
    device: str,
) -> tuple[list[str], float, float]:
    """Generate one request at a time and report aggregate throughput."""
    engine = Engine(model_path, device=device)

    start = time.perf_counter()
    outputs = [engine.generate(prompt, sampling_params) for prompt in prompts]
    elapsed = time.perf_counter() - start

    total_tokens = count_generated_tokens(engine, outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    return outputs, elapsed, throughput


def run_scheduler(
    model_path: str,
    prompts: list[str],
    sampling_params: SamplingParams,
    device: str,
    max_batch_size: int,
) -> tuple[list[str], float, float]:
    """Generate with the scheduler so decode steps are batched together."""
    scheduler = Scheduler(model_path, max_batch_size=max_batch_size, device=device)
    for prompt in prompts:
        scheduler.add_request(prompt, sampling_params)

    start = time.perf_counter()
    outputs = scheduler.run_to_completion(sampling_params)
    elapsed = time.perf_counter() - start

    total_tokens = count_generated_tokens(scheduler.engine, outputs)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0
    return outputs, elapsed, throughput


def benchmark_concurrency(
    model_path: str,
    num_requests: int,
    sampling_params: SamplingParams,
    device: str,
    max_batch_size: int,
    base_prompt: str,
) -> dict[str, float]:
    """Benchmark a single concurrency level for both execution modes."""
    prompts = build_prompts(base_prompt, num_requests)

    _, serial_elapsed, serial_tps = run_serial(model_path, prompts, sampling_params, device)
    _, sched_elapsed, sched_tps = run_scheduler(
        model_path, prompts, sampling_params, device, max_batch_size,
    )

    speedup = sched_tps / serial_tps if serial_tps > 0 else 0.0
    return {
        "requests": num_requests,
        "serial_elapsed": serial_elapsed,
        "serial_tps": serial_tps,
        "scheduler_elapsed": sched_elapsed,
        "scheduler_tps": sched_tps,
        "speedup": speedup,
    }


def print_results(rows: list[dict[str, float]]):
    """Print a compact benchmark table."""
    header = (
        f"{'reqs':>6} {'serial_s':>10} {'serial_tps':>12} "
        f"{'sched_s':>10} {'sched_tps':>12} {'speedup':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{int(row['requests']):>6} "
            f"{row['serial_elapsed']:>10.2f} "
            f"{row['serial_tps']:>12.1f} "
            f"{row['scheduler_elapsed']:>10.2f} "
            f"{row['scheduler_tps']:>12.1f} "
            f"{row['speedup']:>10.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark nano-sglang throughput.")
    parser.add_argument("--model", default=MODEL_NAME, help="Model path or Hugging Face model ID.")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Numbers of concurrent requests to benchmark.",
    )
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum generated tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling threshold.")
    parser.add_argument("--device", default="cuda", help="Torch device to run on.")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Scheduler batch size limit.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Base prompt used for all requests.")
    return parser.parse_args()


def main():
    args = parse_args()
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    rows = []
    for num_requests in args.concurrency:
        rows.append(
            benchmark_concurrency(
                model_path=args.model,
                num_requests=num_requests,
                sampling_params=params,
                device=args.device,
                max_batch_size=args.max_batch_size,
                base_prompt=args.prompt,
            )
        )

    print_results(rows)


if __name__ == "__main__":
    main()
