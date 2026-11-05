/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Embedding Table
 *  Normalized embedding lookup: e_t = (x_t^T W) / ||x_t^T W||_2
 *  All embeddings pre-normalized to unit sphere.
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <math.h>

/* ══════════════════════════════════════════════════════════════
 *  Embedding Create — allocate table from arena
 * ══════════════════════════════════════════════════════════════ */

EmbeddingTable* embedding_create(MemoryArena* arena, uint32_t vocab_size, uint32_t dim) {
    EmbeddingTable* table = (EmbeddingTable*)arena_alloc(arena, sizeof(EmbeddingTable));
    if (!table) return NULL;

    table->vocab_size = vocab_size;
    table->dim        = dim;

    /* Allocate embedding matrix: [vocab_size × dim] */
    table->embeddings = (float*)arena_alloc(arena,
        (size_t)vocab_size * dim * sizeof(float));

    if (!table->embeddings) return NULL;

    return table;
}

/* ══════════════════════════════════════════════════════════════
 *  Embedding Init — random initialization + L2 normalization
 *  Her embedding vektörü birim küre üzerine yerleştirilir.
 * ══════════════════════════════════════════════════════════════ */

void embedding_init_random(EmbeddingTable* table, uint32_t seed) {
    if (!table || !table->embeddings) return;

    uint32_t rng_state[4];
    xoshiro128p_seed(rng_state, seed);

    float std = 1.0f / sqrtf((float)table->dim);

    for (uint32_t v = 0; v < table->vocab_size; v++) {
        float* vec = &table->embeddings[(size_t)v * table->dim];

        /* Random initialization: N(0, 1/sqrt(d)) */
        for (uint32_t d = 0; d < table->dim; d++) {
            vec[d] = random_normal(rng_state) * std;
        }

        /* L2 normalize: ||e||_2 = 1 */
        vec_l2_normalize(vec, table->dim);
    }
}

/* ══════════════════════════════════════════════════════════════
 *  Embedding Lookup
 *  token_id → L2-normalized embedding vector
 *  Şartname §2.1: e_t = (x_t^T W) / ||x_t^T W||_2, ||e_t||_2 = 1
 * ══════════════════════════════════════════════════════════════ */

void embedding_lookup(const EmbeddingTable* table, uint32_t token_id, float* out) {
    if (!table || !out) return;

    /* Bounds check */
    if (token_id >= table->vocab_size) {
        /* Unknown token → zero vector (could also map to <UNK>) */
        vec_zero(out, table->dim);
        return;
    }

    /* Direct copy (embeddings are pre-normalized) */
    vec_copy(out, &table->embeddings[(size_t)token_id * table->dim], table->dim);
}
