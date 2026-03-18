import numpy as np
import ctypes
import os
import subprocess
import sys


def build_shared_library():
    lib_name = "moe.dll"
    cmd = ["gcc", "-O2", "-shared", "-o", lib_name, "moe.c", "-lm"]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)
    return lib_name


def main():
    data = np.load("test_case_shared.npz")

    x = data["x"].astype(np.float32)
    gate_w = data["gate_w"].astype(np.float32)
    experts_w1 = data["experts_w1"].astype(np.float32)
    experts_w2 = data["experts_w2"].astype(np.float32)
    shared_w1 = data["shared_w1"].astype(np.float32)
    shared_w2 = data["shared_w2"].astype(np.float32)

    expected_probs = data["probs"].astype(np.float32)
    expected_topk_idx = data["topk_idx"].astype(np.int32)
    expected_topk_weights = data["topk_weights"].astype(np.float32)
    expected_shared = data["shared_output"].astype(np.float32)
    expected_routed = data["routed_output"].astype(np.float32)
    expected_final = data["final_output"].astype(np.float32)

    lib_path = build_shared_library()
    os.add_dll_directory(os.getcwd())
    lib = ctypes.CDLL(os.path.abspath(lib_path))

    # router_forward
    router_forward = lib.router_forward
    router_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    router_forward.restype = None

    # topk_routing
    topk_routing = lib.topk_routing
    topk_routing.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    topk_routing.restype = None

    # expert_forward
    expert_forward = lib.expert_forward
    expert_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    expert_forward.restype = None

    # moe_forward_shared
    moe_forward_shared = lib.moe_forward_shared
    moe_forward_shared.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_float),  # gate_w
        ctypes.POINTER(ctypes.c_float),  # experts_w1
        ctypes.POINTER(ctypes.c_float),  # experts_w2
        ctypes.POINTER(ctypes.c_float),  # shared_w1
        ctypes.POINTER(ctypes.c_float),  # shared_w2
        ctypes.POINTER(ctypes.c_float),  # routed_out
        ctypes.POINTER(ctypes.c_float),  # shared_out
        ctypes.POINTER(ctypes.c_float),  # final_out
    ]
    moe_forward_shared.restype = None

    # -------- block 1: router --------
    probs = np.zeros(4, dtype=np.float32)
    router_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    print("Expected probs:", expected_probs)
    print("C probs       :", probs)
    router_err = np.max(np.abs(probs - expected_probs))
    print("Router max err:", router_err)

    # -------- block 2: topk --------
    topk_idx = np.zeros(2, dtype=np.int32)
    topk_weights = np.zeros(2, dtype=np.float32)
    topk_routing(
        probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        topk_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        topk_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    print("Expected topk_idx    :", expected_topk_idx)
    print("C topk_idx           :", topk_idx)
    print("Expected topk_weights:", expected_topk_weights)
    print("C topk_weights       :", topk_weights)

    # -------- block 3: shared expert --------
    shared_out = np.zeros_like(x, dtype=np.float32)
    expert_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    print("Expected shared:", expected_shared)
    print("C shared       :", shared_out)
    shared_err = np.max(np.abs(shared_out - expected_shared))
    print("Shared max err :", shared_err)

    # -------- block 4: full moe with shared --------
    routed_out = np.zeros_like(x, dtype=np.float32)
    shared_out2 = np.zeros_like(x, dtype=np.float32)
    final_out = np.zeros_like(x, dtype=np.float32)

    moe_forward_shared(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        experts_w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        experts_w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_w1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        routed_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        shared_out2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        final_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    print("Expected routed:", expected_routed)
    print("C routed       :", routed_out)
    routed_err = np.max(np.abs(routed_out - expected_routed))
    print("Routed max err :", routed_err)

    print("Expected final :", expected_final)
    print("C final        :", final_out)
    final_err = np.max(np.abs(final_out - expected_final))
    print("Final max err  :", final_err)

    ok = True

    if router_err >= 1e-5:
        print("Router test FAILED")
        ok = False

    if not np.array_equal(topk_idx, expected_topk_idx):
        print("TopK index test FAILED")
        ok = False

    if np.max(np.abs(topk_weights - expected_topk_weights)) >= 1e-5:
        print("TopK weight test FAILED")
        ok = False

    if shared_err >= 1e-4:
        print("Shared expert test FAILED")
        ok = False

    if routed_err >= 1e-4:
        print("Routed MoE test FAILED")
        ok = False

    if final_err >= 1e-4:
        print("Final MoE test FAILED")
        ok = False

    if ok:
        print("ALL SHARED-MOE TESTS PASS")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()