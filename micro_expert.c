/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Micro-Expert FFN (Leaf Node Network)
 *  Architecture: input(128) → W_in → hidden(64) → ReLU → W_out → output(128)
 *
 *  Each leaf in the tree contains a small, specialized FFN.
 *  ~65KB per expert (cache-friendly).
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <math.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════
 *  Expert Create — allocate weight matrices from arena
 * ══════════════════════════════════════════════════════════════ */

MicroExpert* expert_create(MemoryArena* arena, uint32_t expert_id) {
    MicroExpert* expert = (MicroExpert*)arena_alloc(arena, sizeof(MicroExpert));
    if (!expert) return NULL;

    expert->expert_id  = expert_id;
    expert->hidden_dim = EXPERT_HIDDEN_DIM;

    /* W_in: [EMBEDDING_DIM × EXPERT_HIDDEN_DIM] = 128 × 64 = 8192 floats = 32 KB */
    expert->W_in = (float*)arena_alloc(arena,
        (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM * sizeof(float));

    /* b_in: [EXPERT_HIDDEN_DIM] = 64 floats = 256 bytes */
    expert->b_in = (float*)arena_alloc(arena,
        (size_t)EXPERT_HIDDEN_DIM * sizeof(float));

    /* W_out: [EXPERT_HIDDEN_DIM × EMBEDDING_DIM] = 64 × 128 = 8192 floats = 32 KB */
    expert->W_out = (float*)arena_alloc(arena,
        (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM * sizeof(float));

    /* b_out: [EMBEDDING_DIM] = 128 floats = 512 bytes */
    expert->b_out = (float*)arena_alloc(arena,
        (size_t)EMBEDDING_DIM * sizeof(float));

    if (!expert->W_in || !expert->b_in || !expert->W_out || !expert->b_out) {
        return NULL;
    }

    return expert;
}

/* ══════════════════════════════════════════════════════════════
 *  Expert Init — Xavier/He random initialization
 *  Uses xoshiro128+ PRNG for deterministic initialization.
 * ══════════════════════════════════════════════════════════════ */

void expert_init_random(MicroExpert* expert, uint32_t seed) {
    if (!expert) return;

    uint32_t rng_state[4];
    xoshiro128p_seed(rng_state, seed + expert->expert_id * 7919);

    /* He initialization: std = sqrt(2 / fan_in) */
    float std_in  = sqrtf(2.0f / (float)EMBEDDING_DIM);
    float std_out = sqrtf(2.0f / (float)EXPERT_HIDDEN_DIM);

    /* Initialize W_in */
    size_t win_size = (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM;
    for (size_t i = 0; i < win_size; i++) {
        expert->W_in[i] = random_normal(rng_state) * std_in;
    }

    /* Initialize b_in (zeros) */
    memset(expert->b_in, 0, EXPERT_HIDDEN_DIM * sizeof(float));

    /* Initialize W_out */
    size_t wout_size = (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM;
    for (size_t i = 0; i < wout_size; i++) {
        expert->W_out[i] = random_normal(rng_state) * std_out;
    }

    /* Initialize b_out (zeros) */
    memset(expert->b_out, 0, EMBEDDING_DIM * sizeof(float));
}

/* ══════════════════════════════════════════════════════════════
 *  Expert Forward Pass
 *  input(128) → matmul(W_in) + b_in → ReLU → matmul(W_out) + b_out → output(128)
 *
 *  Uses AVX-256 accelerated mat_vec_mul and ReLU.
 * ══════════════════════════════════════════════════════════════ */

void expert_forward(const MicroExpert* expert, const float* input, float* output) {
    if (!expert || !input || !output) return;

    /* Temporary buffer for hidden layer (stack allocated, small) */
    alignas(CACHE_LINE_BYTES) float hidden_pre[EXPERT_HIDDEN_DIM];
    alignas(CACHE_LINE_BYTES) float hidden_act[EXPERT_HIDDEN_DIM];

    /* Step 1: hidden_pre = W_in^T · input + b_in
     * W_in is [EMBEDDING_DIM × EXPERT_HIDDEN_DIM], stored row-major
     * We need out[j] = Σ_i W_in[i * EXPERT_HIDDEN_DIM + j] * input[i]
     * This is mat^T · vec, but we store it as mat_vec_mul with M=EXPERT_HIDDEN_DIM, N=EMBEDDING_DIM
     * where mat is the transpose.
     * 
     * Actually, let's store W_in as [EXPERT_HIDDEN_DIM × EMBEDDING_DIM] for efficient matmul.
     * (Convention: each row of W_in corresponds to one hidden unit's input weights)
     */
    mat_vec_mul(expert->W_in, input, expert->b_in,
                hidden_pre, EXPERT_HIDDEN_DIM, EMBEDDING_DIM);

    /* Step 2: ReLU activation */
    vec_relu(hidden_act, hidden_pre, EXPERT_HIDDEN_DIM);

    /* Step 3: output = W_out · hidden_act + b_out
     * W_out is [EMBEDDING_DIM × EXPERT_HIDDEN_DIM] (each row = one output dim)
     */
    mat_vec_mul(expert->W_out, hidden_act, expert->b_out,
                output, EMBEDDING_DIM, EXPERT_HIDDEN_DIM);
}

/* ══════════════════════════════════════════════════════════════
 *  Expert Parameter Count
 * ══════════════════════════════════════════════════════════════ */

size_t expert_param_count(const MicroExpert* expert) {
    if (!expert) return 0;
    /* W_in + b_in + W_out + b_out */
    return (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM  /* W_in  */
         + (size_t)EXPERT_HIDDEN_DIM                   /* b_in  */
         + (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM  /* W_out */
         + (size_t)EMBEDDING_DIM;                      /* b_out */
}
