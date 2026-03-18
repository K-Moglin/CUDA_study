#include <stdio.h>
#include <math.h>

#define HIDDEN 8
#define INTER 16
#define EXPERTS 4
#define TOPK 2

void softmax_inplace(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

float gelu_scalar(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

void expert_forward(
    const float *x,
    const float *w1,
    const float *w2,
    float *out
) {
    float h[INTER];

    // h = gelu(x @ w1)
    for (int j = 0; j < INTER; j++) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN; i++) {
            sum += x[i] * w1[i * INTER + j];
        }
        h[j] = gelu_scalar(sum);
    }

    // out = h @ w2
    for (int j = 0; j < HIDDEN; j++) {
        float sum = 0.0f;
        for (int i = 0; i < INTER; i++) {
            sum += h[i] * w2[i * HIDDEN + j];
        }
        out[j] = sum;
    }
}

void router_forward(
    const float *x,
    const float *gate_w,
    float *probs
) {
    for (int j = 0; j < EXPERTS; j++) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN; i++) {
            sum += x[i] * gate_w[i * EXPERTS + j];
        }
        probs[j] = sum;
    }

    softmax_inplace(probs, EXPERTS);
}

void topk_routing(
    const float *probs,
    int *topk_idx,
    float *topk_weights
) {
    //  Python: np.argsort(probs)[-TOPK:]
    int order[EXPERTS];
    for (int i = 0; i < EXPERTS; i++) {
        order[i] = i;
    }

    // probs sort with indices
    for (int i = 0; i < EXPERTS - 1; i++) {
        for (int j = i + 1; j < EXPERTS; j++) {
            if (probs[order[i]] > probs[order[j]]) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    // last TOPK are the topk indices
    for (int i = 0; i < TOPK; i++) {
        topk_idx[i] = order[EXPERTS - TOPK + i];
        topk_weights[i] = probs[topk_idx[i]];
    }

    float sum = 0.0f;
    for (int i = 0; i < TOPK; i++) {
        sum += topk_weights[i];
    }
    for (int i = 0; i < TOPK; i++) {
        topk_weights[i] /= sum;
    }
}

void moe_forward_shared(
    const float *x,
    const float *gate_w,
    const float *experts_w1,
    const float *experts_w2,
    const float *shared_w1,
    const float *shared_w2,
    float *routed_out,
    float *shared_out,
    float *final_out
) {
    float probs[EXPERTS];
    int topk_idx[TOPK];
    float topk_weights[TOPK];

    router_forward(x, gate_w, probs);
    topk_routing(probs, topk_idx, topk_weights);

    // routed_out = 0
    for (int i = 0; i < HIDDEN; i++) {
        routed_out[i] = 0.0f;
    }

    // routed experts
    for (int t = 0; t < TOPK; t++) {
        int expert_id = topk_idx[t];
        float weight = topk_weights[t];

        const float *w1 = experts_w1 + expert_id * HIDDEN * INTER;
        const float *w2 = experts_w2 + expert_id * INTER * HIDDEN;

        float y[HIDDEN];
        expert_forward(x, w1, w2, y);

        for (int i = 0; i < HIDDEN; i++) {
            routed_out[i] += weight * y[i];
        }
    }

    // shared expert
    expert_forward(x, shared_w1, shared_w2, shared_out);

    // final = routed + shared
    for (int i = 0; i < HIDDEN; i++) {
        final_out[i] = routed_out[i] + shared_out[i];
    }
}

#ifdef TEST_MAIN
int main() {
    printf("moe.c compiled successfully.\n");
    return 0;
}
#endif