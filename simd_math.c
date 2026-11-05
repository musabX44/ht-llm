/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: SIMD Math Implementation
 *  AVX-256 + FMA accelerated vector/matrix operations
 *  with scalar fallbacks for portability.
 * ══════════════════════════════════════════════════════════════════
 */

#include "simd_math.h"
#include <math.h>
#include <float.h>
#include <string.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ══════════════════════════════════════════════════════════════
 *  Dot Product (128-dim optimized)
 * ══════════════════════════════════════════════════════════════ */

float vec_dot_product(const float* a, const float* b, size_t dim) {
#ifdef __AVX2__
    __m256 accum0 = _mm256_setzero_ps();
    __m256 accum1 = _mm256_setzero_ps();

    size_t i = 0;
    /* Unrolled: process 16 floats per iteration (2 × AVX registers) */
    for (; i + 16 <= dim; i += 16) {
        __m256 va0 = _mm256_loadu_ps(&a[i]);
        __m256 vb0 = _mm256_loadu_ps(&b[i]);
        accum0 = _mm256_fmadd_ps(va0, vb0, accum0);

        __m256 va1 = _mm256_loadu_ps(&a[i + 8]);
        __m256 vb1 = _mm256_loadu_ps(&b[i + 8]);
        accum1 = _mm256_fmadd_ps(va1, vb1, accum1);
    }
    /* Handle remaining 8-float blocks */
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        accum0 = _mm256_fmadd_ps(va, vb, accum0);
    }

    /* Merge two accumulators */
    accum0 = _mm256_add_ps(accum0, accum1);

    /* Horizontal sum: 256-bit → 128-bit → scalar */
    __m128 hi128  = _mm256_extractf128_ps(accum0, 1);
    __m128 lo128  = _mm256_castps256_ps128(accum0);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    float result = _mm_cvtss_f32(sums);

    /* Scalar tail: handles dim not a multiple of 8 (safety net for
     * future dimension changes; current config is 8-aligned). */
    for (; i < dim; i++) {
        result += a[i] * b[i];
    }

    return result;
#else
    /* Scalar fallback */
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

/* ══════════════════════════════════════════════════════════════
 *  L2 Norm
 * ══════════════════════════════════════════════════════════════ */

float vec_l2_norm(const float* v, size_t dim) {
    float dot = vec_dot_product(v, v, dim);
    return sqrtf(dot);
}

/* ══════════════════════════════════════════════════════════════
 *  L2 Normalize (in-place). Returns original norm.
 * ══════════════════════════════════════════════════════════════ */

float vec_l2_normalize(float* v, size_t dim) {
    float norm = vec_l2_norm(v, dim);
    if (norm < 1e-12f) return 0.0f;

    float inv_norm = 1.0f / norm;

#ifdef __AVX2__
    __m256 vinv = _mm256_set1_ps(inv_norm);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 val = _mm256_loadu_ps(&v[i]);
        val = _mm256_mul_ps(val, vinv);
        _mm256_storeu_ps(&v[i], val);
    }
    for (; i < dim; i++) {
        v[i] *= inv_norm;
    }
#else
    for (size_t i = 0; i < dim; i++) {
        v[i] *= inv_norm;
    }
#endif

    return norm;
}

/* ══════════════════════════════════════════════════════════════
 *  EMA Update: dst = α·dst + (1-α)·src
 * ══════════════════════════════════════════════════════════════ */

void vec_ema_update(float* dst, const float* src, float alpha, size_t dim) {
    float one_minus_alpha = 1.0f - alpha;

#ifdef __AVX2__
    __m256 valpha = _mm256_set1_ps(alpha);
    __m256 vbeta  = _mm256_set1_ps(one_minus_alpha);

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vd = _mm256_loadu_ps(&dst[i]);
        __m256 vs = _mm256_loadu_ps(&src[i]);
        /* dst = alpha * dst + (1-alpha) * src */
        __m256 result = _mm256_fmadd_ps(valpha, vd, _mm256_mul_ps(vbeta, vs));
        _mm256_storeu_ps(&dst[i], result);
    }
    for (; i < dim; i++) {
        dst[i] = alpha * dst[i] + one_minus_alpha * src[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dst[i] = alpha * dst[i] + one_minus_alpha * src[i];
    }
#endif
}

/* ══════════════════════════════════════════════════════════════
 *  Scaled Vector Add: dst += scale · src
 * ══════════════════════════════════════════════════════════════ */

void vec_add_scaled(float* dst, const float* src, float scale, size_t dim) {
#ifdef __AVX2__
    __m256 vscale = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 vd = _mm256_loadu_ps(&dst[i]);
        __m256 vs = _mm256_loadu_ps(&src[i]);
        vd = _mm256_fmadd_ps(vscale, vs, vd);
        _mm256_storeu_ps(&dst[i], vd);
    }
    for (; i < dim; i++) {
        dst[i] += scale * src[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dst[i] += scale * src[i];
    }
#endif
}

void vec_weighted_add(float* dst, const float* src, float weight, size_t dim) {
    vec_add_scaled(dst, src, weight, dim);
}

/* ══════════════════════════════════════════════════════════════
 *  ReLU: dst[i] = max(0, src[i])
 * ══════════════════════════════════════════════════════════════ */

void vec_relu(float* dst, const float* src, size_t dim) {
#ifdef __AVX2__
    __m256 vzero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(&src[i]);
        v = _mm256_max_ps(v, vzero);
        _mm256_storeu_ps(&dst[i], v);
    }
    for (; i < dim; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
#endif
}

/* ══════════════════════════════════════════════════════════════
 *  Vector Copy / Zero
 * ══════════════════════════════════════════════════════════════ */

void vec_copy(float* dst, const float* src, size_t dim) {
    memcpy(dst, src, dim * sizeof(float));
}

void vec_zero(float* v, size_t dim) {
    memset(v, 0, dim * sizeof(float));
}

/* ══════════════════════════════════════════════════════════════
 *  Matrix-Vector Multiply: out[M] = mat[M×N] · vec[N] + bias[M]
 *  mat is row-major
 * ══════════════════════════════════════════════════════════════ */

void mat_vec_mul(const float* mat, const float* vec, const float* bias,
                 float* out, size_t M, size_t N) {
    for (size_t i = 0; i < M; i++) {
        float dot = vec_dot_product(&mat[i * N], vec, N);
        out[i] = bias ? dot + bias[i] : dot;
    }
}

/* ══════════════════════════════════════════════════════════════
 *  Temperature-Scaled Softmax (numerically stable)
 *  P(i) = exp((x_i - max) / τ) / Σ exp((x_j - max) / τ)
 * ══════════════════════════════════════════════════════════════ */

void vec_softmax(float* values, size_t n, float temperature) {
    if (n == 0) return;

    float inv_temp = 1.0f / temperature;

    /* Find max for numerical stability */
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; i++) {
        float scaled = values[i] * inv_temp;
        if (scaled > max_val) max_val = scaled;
    }

    /* Compute exp(x_i / τ - max) and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(values[i] * inv_temp - max_val);
        values[i] = e;
        sum += e;
    }

    /* Normalize */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < n; i++) {
            values[i] *= inv_sum;
        }
    }
}

/* ══════════════════════════════════════════════════════════════
 *  Shannon Entropy: H = -Σ p_i log(p_i)
 * ══════════════════════════════════════════════════════════════ */

float vec_entropy(const float* probs, size_t n) {
    float H = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (probs[i] > 1e-10f) {
            H -= probs[i] * logf(probs[i]);
        }
    }
    return H;
}

/* ══════════════════════════════════════════════════════════════
 *  Argmax
 * ══════════════════════════════════════════════════════════════ */

size_t vec_argmax(const float* v, size_t n) {
    size_t best = 0;
    float  best_val = v[0];
    for (size_t i = 1; i < n; i++) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

/* ══════════════════════════════════════════════════════════════
 *  Top-k Selection (insertion sort based, efficient for small k)
 * ══════════════════════════════════════════════════════════════ */

int vec_top_k(const float* v, size_t n, int k,
              int* out_indices, float* out_values) {
    if (k <= 0 || n == 0) return 0;
    if ((size_t)k > n) k = (int)n;

    /* Initialize with -inf */
    for (int i = 0; i < k; i++) {
        out_values[i]  = -FLT_MAX;
        out_indices[i] = -1;
    }

    /* Scan and maintain sorted top-k */
    for (size_t i = 0; i < n; i++) {
        if (v[i] > out_values[k - 1]) {
            /* Insert into sorted position */
            int pos = k - 1;
            while (pos > 0 && v[i] > out_values[pos - 1]) {
                out_values[pos]  = out_values[pos - 1];
                out_indices[pos] = out_indices[pos - 1];
                pos--;
            }
            out_values[pos]  = v[i];
            out_indices[pos] = (int)i;
        }
    }

    return k;
}
