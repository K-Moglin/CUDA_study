#include <vector>
#include <cmath>
#include <algorithm>

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void linear(const std::vector<float>& x,
                   const std::vector<float>& w,
                   std::vector<float>& y,
                   int N, int in_dim, int out_dim) {
    y.assign(N * out_dim, 0.0f);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < out_dim; ++c) {
            float s = 0.0f;
            for (int k = 0; k < in_dim; ++k) s += x[r * in_dim + k] * w[k * out_dim + c];
            y[r * out_dim + c] = s;
        }
    }
}

static void expert_forward(const std::vector<float>& x,
                           const std::vector<float>& w1,
                           const std::vector<float>& w2,
                           std::vector<float>& out,
                           int N, int H, int I) {
    std::vector<float> hidden;
    linear(x, w1, hidden, N, H, I);
    for (float& v : hidden) v = gelu(v);
    linear(hidden, w2, out, N, I, H);
}
