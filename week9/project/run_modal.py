import modal
import subprocess
import os
import shutil

app = modal.App("deepseekmoe-week9-tk-main")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "wget", "make")
    .pip_install("torch", "numpy")
    .add_local_file("deepseek_moe_nccl_alltoall.cu", "/root/src/deepseek_moe_nccl_alltoall.cu")
    .add_local_file("generate_assignment_tests.py", "/root/src/generate_assignment_tests.py")
    .add_local_file("check_assignment_tests.py", "/root/src/check_assignment_tests.py")
    .add_local_file("benchmark_baseline_b.py", "/root/src/benchmark_baseline_b.py")
    .add_local_dir("ThunderKittens", "/root/src/ThunderKittens")
)


def trim(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines] + [f"... [truncated {len(lines)-max_lines} lines] ..."])


def summarize_build_output(text: str) -> str:
    keep = []
    for line in text.splitlines():
        low = line.lower()
        if "error:" in low or "warning" in low or line.startswith("nvcc ") or line.startswith("make:"):
            keep.append(line)
    return "\n".join(keep) if keep else "Build finished."


def summarize_check_output(text: str) -> str:
    keep = []
    for line in text.splitlines():
        if (
            "Running:" in line
            or "=> PASS" in line
            or "=> FAIL" in line
            or "All generated assignment test cases passed." in line
            or "TEST PASS" in line
            or "TEST FAIL" in line
            or "max rel error" in line
        ):
            keep.append(line)
    return "\n".join(keep) if keep else trim(text, 60)


def summarize_benchmark_output(text: str) -> str:
    keep = []
    for line in text.splitlines():
        if (
            "===== Baseline B configuration =====" in line
            or "===== Benchmark summary =====" in line
            or "output shape=" in line
            or "output mean=" in line
            or "output std=" in line
            or "PyTorch reference latency" in line
            or "CUDA MoE latency" in line
            or "Speedup" in line
        ):
            keep.append(line)
    return "\n".join(keep) if keep else trim(text, 80)


@app.function(gpu="B200:2", image=image, timeout=60 * 60)
def run_all():
    src_dir = "/root/src"
    work_dir = "/tmp/week9_project"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    for name in [
        "deepseek_moe_nccl_alltoall.cu",
        "generate_assignment_tests.py",
        "check_assignment_tests.py",
        "benchmark_baseline_b.py",
    ]:
        shutil.copy2(os.path.join(src_dir, name), os.path.join(work_dir, name))

    shutil.copytree(
        os.path.join(src_dir, "ThunderKittens"),
        os.path.join(work_dir, "ThunderKittens"),
        dirs_exist_ok=True,
    )
    os.chdir(work_dir)

    all_logs, summary_logs = [], []

    def run_and_collect(cmd, stage, mode="full"):
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        all_logs.append(
            "\n".join([
                f"=== RUN: {' '.join(cmd)} ===",
                result.stdout,
                f"=== EXIT CODE: {result.returncode} ===",
            ])
        )

        if mode == "build":
            shown = summarize_build_output(result.stdout)
        elif mode == "check":
            shown = summarize_check_output(result.stdout)
        elif mode == "benchmark":
            shown = summarize_benchmark_output(result.stdout)
        else:
            shown = trim(result.stdout, 80)

        summary_logs.append(
            "\n".join([
                f"=== {stage.upper()} ===",
                shown.strip(),
                f"status: {'OK' if result.returncode == 0 else 'FAIL'}",
            ])
        )
        return result

    baseline_build = [
        "nvcc", "-O3", "-std=c++17", "deepseek_moe_nccl_alltoall.cu",
        "-o", "deepseek_moe_nccl_alltoall_baseline", "-lnccl"
    ]
    tk_build = [
        "nvcc", "deepseek_moe_nccl_alltoall.cu",
        "-std=c++20", "-O3", "--use_fast_math",
        "-lrt", "-lpthread", "-ldl", "-lcuda", "-lcudadevrt", "-lcudart_static", "-lnccl",
        "--expt-extended-lambda", "--expt-relaxed-constexpr",
        "-forward-unknown-to-host-compiler", "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing",
        "-I./ThunderKittens/include", "-I./ThunderKittens/prototype",
        "-DNDEBUG", "-lineinfo", "-ftemplate-backtrace-limit=0",
        "-DKITTENS_BLACKWELL", "-DUSE_TK_GEMM=1", "-gencode", "arch=compute_100a,code=sm_100a",
        "-o", "deepseek_moe_nccl_alltoall_tk"
    ]

    stages = [
        (["python", "--version"], "python", "full"),
        (["nvcc", "--version"], "nvcc", "build"),
        (baseline_build, "build_baseline", "build"),
        (tk_build, "build_tk_main", "build"),
        (["python", "generate_assignment_tests.py", "--output", "assignment_test_cases.json"], "generate_tests", "full"),
        (["python", "check_assignment_tests.py", "--binary", "./deepseek_moe_nccl_alltoall_tk", "--cases", "assignment_test_cases.json"], "correctness_check_tk", "check"),
        ([
            "python", "benchmark_baseline_b.py", "--device", "cuda", "--dtype", "float16",
            "--batch", "8", "--seq", "128", "--nranks", "2", "--hidden", "1024",
            "--moe-intermediate", "2048", "--experts", "8", "--topk", "2", "--shared", "1",
            "--warmup", "10", "--iters", "50", "--cuda-binary", "./deepseek_moe_nccl_alltoall_baseline",
            "--cuda-args", "--T_local", "512", "--H", "1024", "--I", "2048", "--E", "8",
            "--shared", "1", "--topk", "2", "--warmup", "10", "--iters", "50"
        ], "benchmark_baseline", "benchmark"),
        ([
            "python", "benchmark_baseline_b.py", "--device", "cuda", "--dtype", "float16",
            "--batch", "8", "--seq", "128", "--nranks", "2", "--hidden", "1024",
            "--moe-intermediate", "2048", "--experts", "8", "--topk", "2", "--shared", "1",
            "--warmup", "10", "--iters", "50", "--cuda-binary", "./deepseek_moe_nccl_alltoall_tk",
            "--cuda-args", "--T_local", "512", "--H", "1024", "--I", "2048", "--E", "8",
            "--shared", "1", "--topk", "2", "--warmup", "10", "--iters", "50"
        ], "benchmark_tk_main", "benchmark"),
    ]

    for cmd, stage, mode in stages:
        r = run_and_collect(cmd, stage, mode)
        if r.returncode != 0:
            return {
                "ok": False,
                "stage": stage,
                "summary": "\n\n".join(summary_logs),
                "logs": "\n\n".join(all_logs),
            }

    return {
        "ok": True,
        "stage": "done",
        "summary": "\n\n".join(summary_logs),
        "logs": "\n\n".join(all_logs),
    }


@app.local_entrypoint()
def main():
    result = run_all.remote()
    print(result["summary"])
    print(f"\nFINAL STATUS: {result['ok']} {result['stage']}")
