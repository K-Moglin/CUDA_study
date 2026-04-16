import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def build_if_needed(binary: str, source: str):
    if Path(binary).exists():
        return
    cmd = ["nvcc", "-O3", "-std=c++17", "-o", binary, source, "-lnccl", "-lm"]
    print("Building:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_case(binary: str, case: dict):
    cmd = [
        binary,
        "--check",
        "--T_local", str(case["T_local"]),
        "--H", str(case["H"]),
        "--I", str(case["I"]),
        "--E", str(case["E"]),
        "--shared", str(case["shared"]),
        "--topk", str(case["topk"]),
        "--warmup", str(case.get("warmup", 1)),
        "--iters", str(case.get("iters", 2)),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode == 0 and "TEST PASS" in proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="assignment_test_cases.json")
    parser.add_argument("--binary", default="./deepseek_moe_nccl_alltoall")
    parser.add_argument("--source", default="./deepseek_moe_nccl_alltoall.cu")
    args = parser.parse_args()

    build_if_needed(args.binary, args.source)

    with open(args.cases, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_ok = True
    for case in data["cases"]:
        ok = run_case(args.binary, case)
        print(f"[{case['name']}] => {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    if not all_ok:
        sys.exit(1)

    print("All generated assignment test cases passed.")


if __name__ == "__main__":
    main()
