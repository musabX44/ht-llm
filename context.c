/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Context State Manager
 *  EMA-based running context vector with topic drift detection.
 *
 *  s_ctx,t = α · s_ctx,t-1 + (1-α) · e_t
 *  ||s_ctx||_2 = 1 (always normalized)
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════
 *  Context Init
 * ══════════════════════════════════════════════════════════════ */

void context_init(ContextState* ctx, float alpha) {
    if (!ctx) return;

    vec_zero(ctx->vector, EMBEDDING_DIM);
    ctx->alpha        = alpha;
    ctx->initialized  = 0;
    ctx->token_count  = 0;
    ctx->drift_events = 0;
}

/* ══════════════════════════════════════════════════════════════
 *  Context Update — EMA with continuous normalization
 *
 *  First token: direct copy (no history to blend)
 *  Subsequent: s_ctx = α · s_ctx + (1-α) · e_t, then normalize
 * ══════════════════════════════════════════════════════════════ */

void context_update(ContextState* ctx, const float* embedding) {
    if (!ctx || !embedding) return;

    if (!ctx->initialized) {
        /* İlk token: doğrudan kopyala */
        vec_copy(ctx->vector, embedding, EMBEDDING_DIM);
        ctx->initialized = 1;
    } else {
        /* Topic drift kontrolü */
        int drift = context_detect_topic_drift(ctx, embedding);

        if (drift) {
            /* Ani konu değişimi! α → 0 ile eski bağlamı unut */
            vec_copy(ctx->vector, embedding, EMBEDDING_DIM);
            ctx->drift_events++;
        } else {
            /* Normal EMA güncelleme */
            vec_ema_update(ctx->vector, embedding, ctx->alpha, EMBEDDING_DIM);
        }
    }

    /* Her zaman L2 normalize (||s_ctx||_2 = 1 şartı) */
    vec_l2_normalize(ctx->vector, EMBEDDING_DIM);

    ctx->token_count++;
}

/* ══════════════════════════════════════════════════════════════
 *  Topic Drift Detection
 *  Yeni embedding ile mevcut bağlam arasındaki benzerlik
 *  TOPIC_DRIFT_THRESH (0.15) altına düşerse drift tespit edilir.
 * ══════════════════════════════════════════════════════════════ */

int context_detect_topic_drift(const ContextState* ctx, const float* new_embedding) {
    if (!ctx || !ctx->initialized || !new_embedding) return 0;

    float similarity = vec_dot_product(ctx->vector, new_embedding, EMBEDDING_DIM);

    /* Cosine similarity (vektörler zaten normalized):
     * Düşük benzerlik = farklı konu alanı */
    if (similarity < TOPIC_DRIFT_THRESH) {
        return 1;  /* Topic drift detected */
    }

    return 0;
}

/* ══════════════════════════════════════════════════════════════
 *  Context Reset — manual topic change
 * ══════════════════════════════════════════════════════════════ */

void context_reset(ContextState* ctx) {
    if (!ctx) return;

    vec_zero(ctx->vector, EMBEDDING_DIM);
    ctx->initialized = 0;
    ctx->drift_events++;
}

/* ══════════════════════════════════════════════════════════════
 *  Context Similarity — cosine similarity with current context
 * ══════════════════════════════════════════════════════════════ */

float context_similarity(const ContextState* ctx, const float* embedding) {
    if (!ctx || !ctx->initialized || !embedding) return 0.0f;
    return vec_dot_product(ctx->vector, embedding, EMBEDDING_DIM);
}
