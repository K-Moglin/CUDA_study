import modal

app = modal.App("deepseek-moe-nccl")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .add_local_file("deepseek_moe_modal.cu", "/root/deepseek_moe_modal.cu")
    .add_local_file("reference_moe.cpp", "/root/reference_moe.cpp")
    .add_local_file("README.md", "/root/README.md")
)

@app.function(
    image=image,
    gpu="H100:2",
    cpu=4,
    memory=16384,
    timeout=3600,
)
def run(
    hidden: int = 8,
    intermediate: int = 16,
    tokens: int = 4,
    experts: int = 4,
    topk: int = 2,
    shared: int = 1,
    warmup: int = 3,
    iters: int = 10,
    check: bool = True,
    verbose: bool = True,
):
    import os
    import subprocess

    exe = "/root/deepseek_moe_modal"
    compile_cmd = [
        "nvcc",
        "-O2",
        "-std=c++17",
        "/root/deepseek_moe_modal.cu",
        "-o",
        exe,
        "-lnccl",
    ]
    print("Compiling:", " ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)

    cmd = [
        exe,
        "--hidden", str(hidden),
        "--intermediate", str(intermediate),
        "--tokens", str(tokens),
        "--experts", str(experts),
        "--topk", str(topk),
        "--shared", str(shared),
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    if not check:
        cmd.append("--no-check")
    if not verbose:
        cmd.append("--quiet")

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "INFO"

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


@app.local_entrypoint()
def main(
    hidden: int = 8,
    intermediate: int = 16,
    tokens: int = 4,
    experts: int = 4,
    topk: int = 2,
    shared: int = 1,
    warmup: int = 3,
    iters: int = 10,
    check: bool = True,
    verbose: bool = True,
):
    run.remote(hidden, intermediate, tokens, experts, topk, shared, warmup, iters, check, verbose)
