import numpy as np
from mini_deepseek_moe import moe_forward_with_shared


def main():
    np.random.seed(123)

    hidden = 8
    inter = 16
    experts = 4
    k = 2

    x = np.random.randn(hidden).astype(np.float32)
    gate_w = np.random.randn(hidden, experts).astype(np.float32)
    experts_w1 = np.random.randn(experts, hidden, inter).astype(np.float32)
    experts_w2 = np.random.randn(experts, inter, hidden).astype(np.float32)

    # one shared expert
    shared_w1 = np.random.randn(hidden, inter).astype(np.float32)
    shared_w2 = np.random.randn(inter, hidden).astype(np.float32)

    result = moe_forward_with_shared(
        x,
        gate_w,
        experts_w1,
        experts_w2,
        shared_w1,
        shared_w2,
        k
    )

    np.savez(
        "test_case_shared.npz",
        x=x,
        gate_w=gate_w,
        experts_w1=experts_w1,
        experts_w2=experts_w2,
        shared_w1=shared_w1,
        shared_w2=shared_w2,

        probs=result["probs"].astype(np.float32),
        topk_idx=result["topk_idx"].astype(np.int32),
        topk_weights=result["topk_weights"].astype(np.float32),
        expert_outputs=result["expert_outputs"].astype(np.float32),
        shared_output=result["shared_output"].astype(np.float32),
        routed_output=result["routed_output"].astype(np.float32),
        final_output=result["final_output"].astype(np.float32),

        hidden=np.array([hidden], dtype=np.int32),
        inter=np.array([inter], dtype=np.int32),
        experts=np.array([experts], dtype=np.int32),
        k=np.array([k], dtype=np.int32),
    )

    print("Saved test_case_shared.npz")
    print("shared_output:", result["shared_output"])
    print("routed_output:", result["routed_output"])
    print("final_output:", result["final_output"])


if __name__ == "__main__":
    main()