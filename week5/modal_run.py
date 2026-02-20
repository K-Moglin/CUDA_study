import modal

app = modal.App("week05-cute-fa2")

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git")
    .workdir("/root/proj")
    .add_local_file("main.cu", remote_path="/root/proj/main.cu")
    .add_local_file("kernel.cuh", remote_path="/root/proj/kernel.cuh")
)

@app.function(image=image, gpu="H100", timeout=60 * 20)
def run(n: int = 256, d: int = 64, bc: int = 128, iters: int = 50, causal: int = 1):
    import subprocess

    # Fetch CUTLASS (contains CuTe headers)
    subprocess.run(["bash", "-lc", "rm -rf cutlass && git clone --depth=1 https://github.com/NVIDIA/cutlass.git"], check=True)

    # Show files
    subprocess.run(["bash", "-lc", "ls -la && ls -la cutlass/include/cute | head"], check=True)

    # Compile (CuTe is header-only)
    subprocess.run(
        ["bash", "-lc",
         "nvcc -O3 -std=c++17 -arch=sm_90 "
         "-I./cutlass/include "
         "main.cu -o fa2_cute"],
        check=True,
    )

    # Run
    subprocess.run(["./fa2_cute", str(n), str(d), str(bc), str(iters), str(causal)], check=True)

@app.local_entrypoint()
def main(n: int = 256, d: int = 64, bc: int = 128, iters: int = 50, causal: int = 1):
    run.remote(n, d, bc, iters, causal)
