/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: SIMD Math Library
 *  AVX-256 / FMA accelerated vector operations
 *
 *  Provides both AVX2 and scalar fallback implementations.
 *  Compile with: -mavx2 -mfma for SIMD acceleration.
 * ══════════════════════════════════════════════════════════════════
 */

#ifndef SIMD_MATH_H
#define SIMD_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Vector Operations ── */

/* Dot product of two aligned float vectors */
float vec_dot_product(const float* a, const float* b, size_t dim);

/* L2 norm of a vector */
float vec_l2_norm(const float* v, size_t dim);

/* L2 normalize a vector in-place. Returns original norm. */
float vec_l2_normalize(float* v, size_t dim);

/* EMA update: dst = alpha * dst + (1 - alpha) * src */
void vec_ema_update(float* dst, const float* src, float alpha, size_t dim);

/* Scaled vector add: dst += scale * src */
void vec_add_scaled(float* dst, const float* src, float scale, size_t dim);

/* Weighted accumulate: dst += weight * src (same as vec_add_scaled but semantically different) */
void vec_weighted_add(float* dst, const float* src, float weight, size_t dim);

/* Element-wise ReLU */
void vec_relu(float* dst, const float* src, size_t dim);

/* Copy vector */
void vec_copy(float* dst, const float* src, size_t dim);

/* Zero vector */
void vec_zero(float* v, size_t dim);

/* ── Matrix Operations ── */

/* Matrix-vector multiply: out[M] = mat[M×N] · vec[N] + bias[M]
 * mat is stored row-major: mat[i*N + j] */
void mat_vec_mul(const float* mat, const float* vec, const float* bias,
                 float* out, size_t M, size_t N);

/* ── Statistical Operations ── */

/* Temperature-scaled softmax (numerically stable with max-shift) */
void vec_softmax(float* values, size_t n, float temperature);

/* Shannon entropy: H = -Σ p_i log(p_i), where p_i > 0 */
float vec_entropy(const float* probs, size_t n);

/* Argmax: returns index of maximum value */
size_t vec_argmax(const float* v, size_t n);

/* Top-k selection: fills out_indices and out_values with top-k elements.
 * Returns actual k (may be less if n < k). */
int vec_top_k(const float* v, size_t n, int k,
              int* out_indices, float* out_values);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_MATH_H */
