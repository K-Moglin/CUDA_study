import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    ))

def expert_forward(x, w1, w2):
    """
    x  : (hideden_size)
    w1 : (hidden_size, intermediate)
    w2 : (intermediate, hidden_size) 
    """

    h = gelu(x @ w1)
    y = h @ w2
    return y

def router_forward(x, gate_w):
    """
    x : (hidden_size)
    gate_w : (hidden_size, num_experts)
    """
    scores = x @ gate_w
    probs = softmax(scores)
    return probs

def topk_routing(probs, k):
    """
    probs : (num_experts)
    k : int
    """
    idx = np.argsort(probs)[-k:]
    weight = probs[idx]
    #normalize weight
    weight = weight / np.sum(weight)
    return idx, weight

def moe_forward(
        x,
        gate_w,
        experts_w1,
        experts_w2,
        k
):
    """
    x : (hidden_size)
    gate_w : (hidden_size, num_experts)
    experts_w1 : (num_experts, hidden_size, intermediate)
    experts_w2 : (num_experts, intermediate, hidden_size)
    """
    probs = router_forward(x, gate_w)
    idx, weight = topk_routing(probs, k)
    out = np.zeros_like(x)
    
    for i, w in zip(idx, weight):
        
        y = expert_forward(
            x,
            experts_w1[i],
            experts_w2[i]
        )
        out += w * y
    return out

def moe_forward_with_intermediates(
        x,
        gate_w,
        experts_w1,
        experts_w2,
        k
):
    probs = router_forward(x, gate_w)
    idx, weights = topk_routing(probs, k)

    expert_outputs = []
    out = np.zeros_like(x)

    for i, w in zip(idx, weights):
        y = expert_forward(
            x,
            experts_w1[i],
            experts_w2[i]
        )
        expert_outputs.append(y.copy())
        out += w * y

    expert_outputs = np.stack(expert_outputs, axis=0)

    return {
        "probs": probs,
        "topk_idx": idx,
        "topk_weights": weights,
        "expert_outputs": expert_outputs,
        "final_output": out,
    }

def moe_forward_with_shared(
        x,
        gate_w,
        experts_w1,
        experts_w2,
        shared_w1,
        shared_w2,
        k
):
    probs = router_forward(x, gate_w)
    idx, weights = topk_routing(probs, k)

    routed_part = np.zeros_like(x)
    expert_outputs = []

    for i, w in zip(idx, weights):
        y = expert_forward(
            x,
            experts_w1[i],
            experts_w2[i]
        )
        expert_outputs.append(y.copy())
        routed_part += w * y

    shared_part = expert_forward(x, shared_w1, shared_w2)

    final_output = routed_part + shared_part

    return {
        "probs": probs,
        "topk_idx": idx,
        "topk_weights": weights,
        "expert_outputs": np.stack(expert_outputs, axis=0),
        "shared_output": shared_part,
        "routed_output": routed_part,
        "final_output": final_output,
    }

if __name__ == "__main__":
    np.random.seed(0)

    hidden = 8
    inter = 16
    experts = 4
    k = 2

    x = np.random.rand(hidden)
    gate_w = np.random.rand(hidden, experts)
    experts_w1 = np.random.rand(experts, hidden, inter)
    experts_w2 = np.random.rand(experts, inter, hidden)

    y = moe_forward(
        x,
        gate_w,
        experts_w1,
        experts_w2,
        k
    )

    print("output:", y)