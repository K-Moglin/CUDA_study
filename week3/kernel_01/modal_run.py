import modal

app = modal.App("kernel_01_naive")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.3.2-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
    .workdir("/root/proj")
    .add_local_dir(".", remote_path="/root/proj")
)

@app.function(image=image, gpu="H100", timeout=60 * 20)
def run(m: int = 4096, n: int = 4096, k: int = 4096, iters: int = 50):
    import subprocess
    subprocess.run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "main.cu", "-o", "k1"],
        check=True,
    )
    subprocess.run(["./k1", str(m), str(n), str(k), str(iters)], check=True)

@app.local_entrypoint()
def main(m: int = 4096, n: int = 4096, k: int = 4096, iters: int = 50):
    run.remote(m, n, k, iters)

