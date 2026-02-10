import modal

app = modal.App("FlashAttention2 CUDA")

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential")
    .workdir("/root/proj")
    .add_local_file("main.cu", remote_path="/root/proj/main.cu")
    .add_local_file("kernel.cuh", remote_path="/root/proj/kernel.cuh")
)

@app.function(image=image, gpu="H100", timeout=60 * 20)
def run(n: int = 256, d: int = 64, bc: int = 128, iters: int = 50, causal: int = 1):
    import subprocess
    subprocess.run(["bash", "-lc", "ls -la && wc -c main.cu kernel.cuh && head -n 5 main.cu"], check=True)

    # compile
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "main.cu", "-o", "fa2_cuda"],
        check=True,
    )

    # run
    subprocess.run(["./fa2_cuda", str(n), str(d), str(bc), str(iters), str(causal)], check=True)

@app.local_entrypoint()
def main(n: int = 256, d: int = 64, bc: int = 128, iters: int = 50, causal: int = 1):
    run.remote(n, d, bc, iters, causal)
