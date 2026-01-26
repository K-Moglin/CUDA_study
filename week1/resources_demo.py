import os
import modal

app = modal.App("resources-demo")

# 1) Basic resource reservation:
#    - cpu is the number of physical CPU cores
#    - memory is specified in MiB
@app.function(cpu=2.0, memory=2048)
def basic_reservations():
    return {
        # This shows how many CPUs the OS reports as visible.
        # Note: this is NOT guaranteed to equal the reserved CPU count.
        "cpu_count_seen_by_os": os.cpu_count(),
        "note": "Resource reservation guarantees a minimum, not a fixed value.",
    }

# 2) Resource request + limit
#    The function requests at least 1 CPU and 1 GiB memory,
#    but can scale up to 4 CPUs and 2 GiB memory if available.
@app.function(cpu=(1.0, 4.0), memory=(1024, 2048))
def with_limits():
    return "ok"

# 3) Request additional ephemeral disk space
#    - ephemeral_disk is specified in MiB
#    - Maximum allowed value is 3,145,728 MiB (~3.0 TiB)
@app.function()
def disk_demo():
    path = "/tmp/hello.txt"
    with open(path, "w") as f:
        f.write("hello disk\n")
    return {
        "wrote": path,
        "size_bytes": os.path.getsize(path),
    }

# Local entrypoint used when running with `modal run`
@app.local_entrypoint()
def main():
    print("basic_reservations:", basic_reservations.remote())
    print("with_limits:", with_limits.remote())
    print("disk_demo:", disk_demo.remote())
