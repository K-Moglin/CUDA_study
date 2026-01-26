import modal

# Choose a CUDA devel image (includes nvcc + toolchain)
CUDA_VERSION = "12.8.1"
OS = "ubuntu24.04"
TAG = f"{CUDA_VERSION}-devel-{OS}"

app = modal.App("cuda-gemm")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{TAG}", add_python="3.12")
    .entrypoint([])                 # reduce noisy base entrypoint output
    .apt_install("build-essential") # g++/make (nvcc already present)
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .run_commands("nvcc -O3 -std=c++17 /root/src/gemm.cu -o /root/gemm_test")
)

@app.function(gpu="any", image=image, timeout=60 * 10)
def run_tests():
    import subprocess
    out = subprocess.check_output(["/root/gemm_test", "--test"], text=True)
    return out

@app.function(gpu="any", image=image, timeout=60 * 10)
def run_bench(m: int, n: int, k: int, transA: int, transB: int, iters: int = 50):
    import subprocess
    out = subprocess.check_output(
        ["/root/gemm_test", "--bench", str(m), str(n), str(k), str(transA), str(transB), str(iters)],
        text=True,
    )
    return out

@app.local_entrypoint()
def main():
    print(run_tests.remote())

    # Example benchmark (increase sizes to make it measurable)
    print(run_bench.remote(2048, 2048, 2048, 0, 0, 20))
    print(run_bench.remote(2048, 2048, 2048, 1, 0, 20))
    print(run_bench.remote(2048, 2048, 2048, 0, 1, 20))
    print(run_bench.remote(2048, 2048, 2048, 1, 1, 20))
