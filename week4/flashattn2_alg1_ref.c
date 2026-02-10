#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static inline float fmaxf2(float a, float b) { return a > b ? a : b; }

// Q,K,V: [N][d] row-major
// O: [N][d], L: [N]
// Br,Bc tiling, optional causal mask
void flashattention2_alg1_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    size_t N, size_t d,
    size_t Br, size_t Bc,
    int causal
) {
  const float inv_sqrt_d = 1.0f / sqrtf((float)d);

  for (size_t i0 = 0; i0 < N; i0 += Br) {
    const size_t i_max = (i0 + Br < N) ? (i0 + Br) : N;
    const size_t br = i_max - i0;

    float* m_local = (float*)malloc(br * sizeof(float));
    float* l_local = (float*)malloc(br * sizeof(float));
    float* Otilde  = (float*)malloc(br * d * sizeof(float));
    if (!m_local || !l_local || !Otilde) {
      fprintf(stderr, "malloc failed\n");
      free(m_local); free(l_local); free(Otilde);
      return;
    }

    for (size_t r = 0; r < br; ++r) {
      m_local[r] = -INFINITY;
      l_local[r] = 0.0f;
    }
    memset(Otilde, 0, br * d * sizeof(float));

    for (size_t j0 = 0; j0 < N; j0 += Bc) {
      const size_t j_max = (j0 + Bc < N) ? (j0 + Bc) : N;
      const size_t bc = j_max - j0;

      // safe skip for causal: if this whole K/V block is strictly future for all rows in this Q block
      if (causal) {
        size_t last_row = i0 + br - 1;
        if (j0 > last_row) break;
      }

      for (size_t r = 0; r < br; ++r) {
        const size_t i = i0 + r;

        // tile rowmax over [j0, j_max)
        float tile_rowmax = -INFINITY;
        const float* Qi = Q + i * d;

        for (size_t cj = 0; cj < bc; ++cj) {
          const size_t j = j0 + cj;
          if (causal && j > i) continue;

          const float* Kj = K + j * d;
          float s = 0.0f;
          for (size_t k = 0; k < d; ++k) s += Qi[k] * Kj[k];
          s *= inv_sqrt_d;
          tile_rowmax = fmaxf2(tile_rowmax, s);
        }

        if (!isfinite(tile_rowmax)) {
          // this tile contributes nothing for this row (fully masked)
          continue;
        }

        const float m_old = m_local[r];
        const float m_new = isfinite(m_old) ? fmaxf2(m_old, tile_rowmax) : tile_rowmax;

        // rescale old accumulators to new base
        const float scale_old = isfinite(m_old) ? expf(m_old - m_new) : 0.0f;

        float* Otilde_row = Otilde + r * d;
        if (scale_old != 1.0f) {
          for (size_t k = 0; k < d; ++k) Otilde_row[k] *= scale_old;
          l_local[r] *= scale_old;
        }

        // accumulate tile contributions under base m_new
        float tile_l = 0.0f;

        for (size_t cj = 0; cj < bc; ++cj) {
          const size_t j = j0 + cj;
          if (causal && j > i) continue;

          const float* Kj = K + j * d;
          float s = 0.0f;
          for (size_t k = 0; k < d; ++k) s += Qi[k] * Kj[k];
          s *= inv_sqrt_d;

          const float p = expf(s - m_new); // P~ = exp(score - m_new)
          tile_l += p;

          const float* Vj = V + j * d;
          for (size_t k = 0; k < d; ++k) Otilde_row[k] += p * Vj[k];
        }

        l_local[r] += tile_l;
        m_local[r] = m_new;
      }
    }

    // finalize block: O = Otilde / l, L = m + log(l)
    for (size_t r = 0; r < br; ++r) {
      const size_t i = i0 + r;
      float* Oi = O + i * d;

      const float mi = m_local[r];
      const float li = l_local[r];

      if (!isfinite(mi) || !(li > 0.0f) || !isfinite(li)) {
        for (size_t k = 0; k < d; ++k) Oi[k] = 0.0f;
        L[i] = -INFINITY;
        continue;
      }

      const float inv_li = 1.0f / li;
      const float* Otilde_row = Otilde + r * d;
      for (size_t k = 0; k < d; ++k) Oi[k] = Otilde_row[k] * inv_li;
      L[i] = mi + logf(li);
    }

    free(m_local);
    free(l_local);
    free(Otilde);
  }
}

// ---- Naive reference: full softmax(QK^T) V ----
static void naive_attention(
    const float* Q, const float* K, const float* V,
    float* O,
    size_t N, size_t d,
    int causal
) {
  const float inv_sqrt_d = 1.0f / sqrtf((float)d);

  float* scores = (float*)malloc(N * sizeof(float));
  if (!scores) return;

  for (size_t i = 0; i < N; ++i) {
    // compute scores
    float m = -INFINITY;
    for (size_t j = 0; j < N; ++j) {
      if (causal && j > i) {
        scores[j] = -INFINITY;
        continue;
      }
      float s = 0.0f;
      for (size_t k = 0; k < d; ++k) s += Q[i*d+k] * K[j*d+k];
      s *= inv_sqrt_d;
      scores[j] = s;
      m = fmaxf2(m, s);
    }
    // softmax denom
    float l = 0.0f;
    for (size_t j = 0; j < N; ++j) {
      if (!isfinite(scores[j])) continue;
      l += expf(scores[j] - m);
    }
    // O[i] = sum softmax * V
    for (size_t k = 0; k < d; ++k) O[i*d+k] = 0.0f;
    if (l > 0.0f) {
      for (size_t j = 0; j < N; ++j) {
        if (!isfinite(scores[j])) continue;
        float p = expf(scores[j] - m) / l;
        for (size_t k = 0; k < d; ++k) O[i*d+k] += p * V[j*d+k];
      }
    }
  }

  free(scores);
}

static float frand01(void) { return (float)rand() / (float)RAND_MAX; }

int main(void) {
  // small test
  const size_t N = 64;
  const size_t d = 32;
  const size_t Br = 16;
  const size_t Bc = 16;
  const int causal = 1;

  float* Q = (float*)malloc(N*d*sizeof(float));
  float* K = (float*)malloc(N*d*sizeof(float));
  float* V = (float*)malloc(N*d*sizeof(float));
  float* O1 = (float*)malloc(N*d*sizeof(float));
  float* O2 = (float*)malloc(N*d*sizeof(float));
  float* L = (float*)malloc(N*sizeof(float));
  if (!Q||!K||!V||!O1||!O2||!L) return 1;

  srand(0);
  for (size_t i = 0; i < N*d; ++i) {
    Q[i] = frand01()*2.0f - 1.0f;
    K[i] = frand01()*2.0f - 1.0f;
    V[i] = frand01()*2.0f - 1.0f;
  }

  flashattention2_alg1_forward(Q, K, V, O1, L, N, d, Br, Bc, causal);
  naive_attention(Q, K, V, O2, N, d, causal);

  // compare max abs error
  float max_err = 0.0f;
  for (size_t i = 0; i < N*d; ++i) {
    float e = fabsf(O1[i] - O2[i]);
    if (e > max_err) max_err = e;
  }
  printf("max |O_flash - O_naive| = %.8f\n", max_err);
  printf("Example L[0]=%f L[N-1]=%f\n", L[0], L[N-1]);

  free(Q); free(K); free(V); free(O1); free(O2); free(L);
  return 0;
}
