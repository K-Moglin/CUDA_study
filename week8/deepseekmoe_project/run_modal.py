import modal
import re

app = modal.App("deepseekmoe-week8")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "wget")
    .pip_install("torch", "numpy")
    .add_local_dir(".", "/root/project")
)


def summarize_check_output(text: str) -> str:
    lines = text.splitlines()
    keep = []

    for line in lines:
        if (
            "Running:" in line
            or "=> PASS" in line
            or "=> FAIL" in line
            or "All generated assignment test cases passed." in line
            or "TEST PASS" in line
            or "TEST FAIL" in line
        ):
            keep.append(line)

    if not keep:
        return text
    return "\n".join(keep)


def summarize_benchmark_output(text: str) -> str:
    lines = text.splitlines()
    keep = []
    in_config = False
    in_summary = False

    for line in lines:
        if "===== Baseline B configuration =====" in line:
            in_config = True
            in_summary = False
            keep.append(line)
            continue
        if "===== Benchmark summary =====" in line:
            in_summary = True
            in_config = False
            keep.append("")
            keep.append(line)
            continue
        if "===== Sample output stats =====" in line:
            in_summary = False
            keep.append("")
            keep.append(line)
            continue

        if in_config:
            if line.strip() == "":
                in_config = False
            else:
                keep.append(line)
        elif in_summary:
            keep.append(line)
        elif line.startswith("output shape=") or line.startswith("output mean=") or line.startswith("output std="):
            keep.append(line)

    if not keep:
        return text
    return "\n".join(keep)


@app.function(gpu="H100:2", image=image, timeout=60 * 60)
def run_all():
    import os
    import subprocess

    os.chdir("/root/project")
    all_logs = []
    summary_logs = []

    def run_and_collect(cmd, stage, mode="full"):
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        raw_block = "\n".join(
            [
                f"=== RUN: {' '.join(cmd)} ===",
                result.stdout,
                f"=== EXIT CODE: {result.returncode} ===",
            ]
        )
        all_logs.append(raw_block)

        if mode == "check":
            shown = summarize_check_output(result.stdout)
        elif mode == "benchmark":
            shown = summarize_benchmark_output(result.stdout)
        else:
            shown = result.stdout

        short_block = "\n".join(
            [
                f"=== {stage.upper()} ===",
                shown.strip(),
                f"status: {'OK' if result.returncode == 0 else 'FAIL'}",
            ]
        )
        summary_logs.append(short_block)

        return result

    r = run_and_collect(["python", "--version"], "python", "full")
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "python_version",
            "summary": "\n\n".join(summary_logs),
            "logs": "\n\n".join(all_logs),
        }

    r = run_and_collect(["nvcc", "--version"], "nvcc", "full")
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "nvcc_version",
            "summary": "\n\n".join(summary_logs),
            "logs": "\n\n".join(all_logs),
        }

    r = run_and_collect(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-o",
            "deepseek_moe_nccl_alltoall",
            "deepseek_moe_nccl_alltoall.cu",
            "-lnccl",
        ],
        "build",
        "full",
    )
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "build",
            "summary": "\n\n".join(summary_logs),
            "logs": "\n\n".join(all_logs),
        }

    r = run_and_collect(
        ["python", "generate_assignment_tests.py", "--output", "assignment_test_cases.json"],
        "generate_tests",
        "full",
    )
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "generate_tests",
            "summary": "\n\n".join(summary_logs),
            "logs": "\n\n".join(all_logs),
        }

    r = run_and_collect(
        [
            "python",
            "check_assignment_tests.py",
            "--binary",
            "./deepseek_moe_nccl_alltoall",
            "--cases",
            "assignment_test_cases.json",
        ],
        "correctness_check",
        "check",
    )
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "check",
            "summary": "\n\n".join(summary_logs),
            "logs": "\n\n".join(all_logs),
        }

    r = run_and_collect(
        [
            "python",
            "benchmark_baseline_b.py",
            "--device", "cuda",
            "--dtype", "float16",
            "--batch", "8",
            "--seq", "128",
            "--nranks", "2",
            "--hidden", "1024",
            "--moe-intermediate", "2048",
            "--experts", "8",
            "--topk", "2",
            "--shared", "1",
            "--warmup", "10",
            "--iters", "50",
            "--cuda-binary", "./deepseek_moe_nccl_alltoall",
            "--cuda-args",
            "--T_local", "512",
            "--H", "1024",
            "--I", "2048",
            "--E", "8",
            "--shared", "1",
            "--topk", "2",
            "--warmup", "10",
            "--iters", "50",
        ],
        "benchmark",
        "benchmark",
    )
    if r.returncode != 0:
        return {
            "ok": False,
            "stage": "benchmark",
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

    if not result["ok"]:
        print("\n========== FULL FAILURE LOG ==========")
        print(result["logs"])