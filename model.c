/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Model API (Top-level interface)
 *
 *  htllm_create()  → Build model in arena
 *  htllm_forward() → embed → context → route → expert → output
 *  htllm_generate() → autoregressive token generation
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ══════════════════════════════════════════════════════════════
 *  Utility: PRNG (xoshiro128+)
 *  Hızlı, deterministik, düşük footprint PRNG.
 * ══════════════════════════════════════════════════════════════ */

static inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint32_t xoshiro128p_next(uint32_t state[4]) {
    uint32_t result = state[0] + state[3];
    uint32_t t = state[1] << 9;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotl32(state[3], 11);

    return result;
}

void xoshiro128p_seed(uint32_t state[4], uint32_t seed) {
    /* SplitMix32 to initialize state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B9u;
        uint32_t z = seed;
        z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
        z = (z ^ (z >> 13)) * 0xC2B2AE35u;
        z = z ^ (z >> 16);
        state[i] = z;
    }
}

float random_uniform(uint32_t state[4]) {
    return (float)(xoshiro128p_next(state) >> 8) / 16777216.0f;
}

float random_normal(uint32_t state[4]) {
    /* Box-Muller transform */
    float u1 = random_uniform(state);
    float u2 = random_uniform(state);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ══════════════════════════════════════════════════════════════
 *  Utility: High-resolution timer
 * ══════════════════════════════════════════════════════════════ */

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/* ══════════════════════════════════════════════════════════════
 *  Output Head Create & Forward
 *  Shared projection: embedding_dim(128) → vocab_size(4096)
 * ══════════════════════════════════════════════════════════════ */

OutputHead* output_head_create(MemoryArena* arena) {
    OutputHead* head = (OutputHead*)arena_alloc(arena, sizeof(OutputHead));
    if (!head) return NULL;

    /* W_proj: [EMBEDDING_DIM × VOCAB_SIZE] */
    head->W_proj = (float*)arena_alloc(arena,
        (size_t)EMBEDDING_DIM * VOCAB_SIZE * sizeof(float));

    /* b_proj: [VOCAB_SIZE] */
    head->b_proj = (float*)arena_alloc(arena,
        (size_t)VOCAB_SIZE * sizeof(float));

    if (!head->W_proj || !head->b_proj) return NULL;

    return head;
}

void output_head_init_random(OutputHead* head, uint32_t seed) {
    if (!head) return;

    uint32_t rng_state[4];
    xoshiro128p_seed(rng_state, seed);

    float std = 1.0f / sqrtf((float)EMBEDDING_DIM);

    size_t proj_size = (size_t)EMBEDDING_DIM * VOCAB_SIZE;
    for (size_t i = 0; i < proj_size; i++) {
        head->W_proj[i] = random_normal(rng_state) * std;
    }

    memset(head->b_proj, 0, VOCAB_SIZE * sizeof(float));
}

void output_head_forward(const OutputHead* head, const float* expert_out, float* logits) {
    if (!head || !expert_out || !logits) return;

    /* logits[v] = Σ_d W_proj[d * VOCAB_SIZE + v] * expert_out[d] + b_proj[v]
     * Rewritten for row-major: W_proj is [EMBEDDING_DIM × VOCAB_SIZE]
     * logits = expert_out^T · W_proj + b_proj
     */
    /* Initialize with bias */
    vec_copy(logits, head->b_proj, VOCAB_SIZE);

    /* Accumulate: for each embedding dimension */
    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
        float scale = expert_out[d];
        if (fabsf(scale) < 1e-8f) continue;  /* Skip near-zero */
        vec_add_scaled(logits, &head->W_proj[d * VOCAB_SIZE], scale, VOCAB_SIZE);
    }
}

/* ══════════════════════════════════════════════════════════════
 *  HT-LLM Create — build entire model in a single arena
 * ══════════════════════════════════════════════════════════════ */

HTLLMModel* htllm_create(uint32_t seed) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   HT-LLM: Hierarchical Tree Language Model  v1.0             ║\n");
    printf("║   CPU-Focused Sparse-Activation Language Model Prototype     ║\n");
    printf("║   Tasarımcı: Musab Küçük                                     ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║   Embedding Dim: %d  │  Vocab: %d  │  B: %d               ║\n",
           EMBEDDING_DIM, VOCAB_SIZE, MAX_CHILDREN);
    printf("║   Derinlik: %d       │  Yaprak: %d │  Uzman Gizli: %d     ║\n",
           TREE_DEPTH, NUM_LEAF_NODES, EXPERT_HIDDEN_DIM);
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");

    /* Calculate memory requirements */
    size_t arena_size = 0;
    arena_size += sizeof(HTLLMModel);                                                  /* Model struct */
    arena_size += TOTAL_NODES * sizeof(TreeNode);                                      /* Tree nodes */
    arena_size += NUM_LEAF_NODES * sizeof(MicroExpert);                               /* Expert structs */
    arena_size += NUM_LEAF_NODES * (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM * sizeof(float);  /* W_in */
    arena_size += NUM_LEAF_NODES * (size_t)EXPERT_HIDDEN_DIM * sizeof(float);                  /* b_in */
    arena_size += NUM_LEAF_NODES * (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM * sizeof(float);  /* W_out */
    arena_size += NUM_LEAF_NODES * (size_t)EMBEDDING_DIM * sizeof(float);                      /* b_out */
    arena_size += sizeof(EmbeddingTable);
    arena_size += (size_t)VOCAB_SIZE * EMBEDDING_DIM * sizeof(float);                  /* Embeddings */
    arena_size += sizeof(OutputHead);
    arena_size += (size_t)EMBEDDING_DIM * VOCAB_SIZE * sizeof(float);                  /* W_proj */
    arena_size += (size_t)VOCAB_SIZE * sizeof(float);                                   /* b_proj */
    arena_size += sizeof(Tokenizer) + (size_t)VOCAB_SIZE * (sizeof(char*) + 32);   /* Tokenizer (dinamik vocab, ort. 32 byte/kelime) */
    arena_size += arena_size / 4;  /* 25% overhead for alignment padding */

    printf("  [Bellek] Hesaplanan arena boyutu: %.2f MB\n",
           (double)arena_size / (1024.0 * 1024.0));

    /* Create arena */
    MemoryArena* arena = arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "HATA: Arena oluşturulamadı!\n");
        return NULL;
    }

    /* Allocate model struct */
    HTLLMModel* model = (HTLLMModel*)arena_alloc(arena, sizeof(HTLLMModel));
    if (!model) {
        arena_destroy(arena);
        return NULL;
    }
    memset(model, 0, sizeof(HTLLMModel));
    model->arena = arena;

    /* ── Build components ── */
    printf("  [Model] Bileşenler oluşturuluyor...\n");

    /* 1. Tokenizer */
    printf("    → Tokenizer...\n");
    model->tokenizer = tokenizer_create(arena, VOCAB_SIZE);

    /* 2. Embedding table */
    printf("    → Embedding tablosu (%d × %d)...\n", VOCAB_SIZE, EMBEDDING_DIM);
    model->embeddings = embedding_create(arena, VOCAB_SIZE, EMBEDDING_DIM);
    if (model->embeddings) {
        embedding_init_random(model->embeddings, seed);
    }

    /* 3. Output head */
    printf("    → Çıktı projeksiyon katmanı (%d × %d)...\n", EMBEDDING_DIM, VOCAB_SIZE);
    model->output_head = output_head_create(arena);
    if (model->output_head) {
        output_head_init_random(model->output_head, seed + 999);
    }

    /* 4. Tree with micro-experts */
    printf("    → Hiyerarşik ağaç ve mikro-uzmanlar...\n");
    model->root = tree_create_node(arena, 0, 0);
    if (model->root) {
        tree_build_random(arena, model->root, seed);
    }

    /* 5. Context state */
    context_init(&model->context, EMA_ALPHA_DEFAULT);

    /* ── Compute statistics ── */
    model->total_nodes = tree_count_nodes(model->root);
    model->leaf_nodes  = tree_count_leaves(model->root);

    /* Total parameters */
    model->total_params = 0;
    model->total_params += (uint64_t)VOCAB_SIZE * EMBEDDING_DIM;   /* Embeddings */
    model->total_params += (uint64_t)EMBEDDING_DIM * VOCAB_SIZE;   /* Output proj */
    model->total_params += (uint64_t)VOCAB_SIZE;                    /* Output bias */
    model->total_params += (uint64_t)model->leaf_nodes * (
        (uint64_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM +              /* W_in per expert */
        (uint64_t)EXPERT_HIDDEN_DIM +                               /* b_in */
        (uint64_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM +              /* W_out */
        (uint64_t)EMBEDDING_DIM                                     /* b_out */
    );
    model->total_params += (uint64_t)model->total_nodes * EMBEDDING_DIM;  /* Centroids */

    model->model_memory_bytes = arena->offset;

    printf("\n  [Model] Oluşturma tamamlandı!\n");

    return model;
}

/* ══════════════════════════════════════════════════════════════
 *  Forward Pass — Single token inference
 *
 *  Pipeline:
 *  1. Embedding lookup (L2 normalized)
 *  2. Context vector update (EMA)
 *  3. Hierarchical tree routing (O(log N))
 *  4. Micro-expert forward pass (selected leaves only)
 *  5. Weighted output merge
 *  6. Output head projection → logits
 * ══════════════════════════════════════════════════════════════ */

void htllm_forward(HTLLMModel* model, uint32_t token_id, float* logits) {
    if (!model || !logits) return;

    /* Step 1: Embedding lookup */
    alignas(CACHE_LINE_BYTES) float embedding[EMBEDDING_DIM];
    embedding_lookup(model->embeddings, token_id, embedding);

    /* Step 2: Update context vector */
    context_update(&model->context, embedding);

    /* Step 3: Route through tree */
    TreeNode* selected_leaves[TOP_K_MAX * TOP_K_MAX];
    float     leaf_weights[TOP_K_MAX * TOP_K_MAX];
    int       num_leaves = 0;

    tree_route_full(model->root, model->context.vector,
                    selected_leaves, leaf_weights, &num_leaves);

    /* Step 4 & 5: Expert forward + weighted merge */
    alignas(CACHE_LINE_BYTES) float merged_output[EMBEDDING_DIM];
    vec_zero(merged_output, EMBEDDING_DIM);

    uint64_t active_params = 0;

    for (int i = 0; i < num_leaves; i++) {
        alignas(CACHE_LINE_BYTES) float expert_out[EMBEDDING_DIM];

        expert_forward(selected_leaves[i]->expert,
                      model->context.vector, expert_out);

        /* Weighted accumulate */
        vec_weighted_add(merged_output, expert_out, leaf_weights[i], EMBEDDING_DIM);

        active_params += expert_param_count(selected_leaves[i]->expert);
    }

    /* Track active parameters */
    active_params += EMBEDDING_DIM;                            /* Current embedding */
    active_params += EMBEDDING_DIM * VOCAB_SIZE + VOCAB_SIZE;  /* Output head */
    model->active_params_last = active_params;

    /* Step 6: Output head → logits */
    output_head_forward(model->output_head, merged_output, logits);
}

/* ══════════════════════════════════════════════════════════════
 *  Generate Next Token — greedy argmax
 * ══════════════════════════════════════════════════════════════ */

uint32_t htllm_generate_next(HTLLMModel* model, uint32_t token_id) {
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    if (!logits) return 1;  /* <unk> */

    htllm_forward(model, token_id, logits);

    /* Apply softmax */
    vec_softmax(logits, VOCAB_SIZE, 1.0f);

    /* Greedy argmax */
    uint32_t best = (uint32_t)vec_argmax(logits, VOCAB_SIZE);

    free(logits);
    return best;
}

/* ══════════════════════════════════════════════════════════════
 *  Generate — autoregressive sequence generation
 * ══════════════════════════════════════════════════════════════ */

void htllm_generate(HTLLMModel* model, const char* prompt, int max_tokens) {
    if (!model || !prompt) return;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║              Token Üretimi (Generation)                ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Girdi: \"%s\"\n", prompt);
    printf("  ║  Maks. token: %d\n", max_tokens);
    printf("  ╠══════════════════════════════════════════════════════════╣\n");

    /* Encode prompt */
    uint32_t token_ids[256];
    int n_tokens = tokenizer_encode(model->tokenizer, prompt, token_ids, 256);

    printf("  ║  Kodlanmış tokenlar (%d): ", n_tokens);
    for (int i = 0; i < n_tokens && i < 10; i++) {
        printf("[%u:%s] ", token_ids[i], tokenizer_decode(model->tokenizer, token_ids[i]));
    }
    if (n_tokens > 10) printf("...");
    printf("\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");

    /* Process prompt tokens */
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    if (!logits) return;

    double total_time = 0.0;
    uint32_t last_token = token_ids[0];

    for (int i = 0; i < n_tokens; i++) {
        double t0 = get_time_ms();
        htllm_forward(model, token_ids[i], logits);
        double t1 = get_time_ms();
        total_time += (t1 - t0);
        last_token = token_ids[i];
    }

    printf("  ║  Prompt işleme: %.2f ms (%.2f ms/token)\n",
           total_time, n_tokens > 0 ? total_time / n_tokens : 0);

    /* Generate new tokens */
    printf("  ║  Üretilen: ");
    double gen_time = 0.0;

    for (int t = 0; t < max_tokens; t++) {
        double t0 = get_time_ms();
        htllm_forward(model, last_token, logits);
        double t1 = get_time_ms();
        gen_time += (t1 - t0);

        vec_softmax(logits, VOCAB_SIZE, 1.0f);
        last_token = (uint32_t)vec_argmax(logits, VOCAB_SIZE);

        /* Stop on EOS */
        if (last_token == 3) break;

        printf("%s ", tokenizer_decode(model->tokenizer, last_token));
    }
    printf("\n");

    double avg_ms = max_tokens > 0 ? gen_time / max_tokens : 0;
    double tokens_per_sec = avg_ms > 0 ? 1000.0 / avg_ms : 0;

    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Üretim süresi: %.2f ms toplam\n", gen_time);
    printf("  ║  Ortalama latency: %.2f ms/token\n", avg_ms);
    printf("  ║  Hız: %.1f token/saniye\n", tokens_per_sec);
    printf("  ║  Aktif parametre: %lu / %lu (%%%.2f)\n",
           (unsigned long)model->active_params_last,
           (unsigned long)model->total_params,
           model->total_params > 0 ?
               100.0 * (double)model->active_params_last / (double)model->total_params : 0);
    printf("  ╚══════════════════════════════════════════════════════════╝\n");

    free(logits);
}

/* ══════════════════════════════════════════════════════════════
 *  Print Model Statistics
 * ══════════════════════════════════════════════════════════════ */

void htllm_print_stats(const HTLLMModel* model) {
    if (!model) return;

    double total_mb = (double)model->model_memory_bytes / (1024.0 * 1024.0);

    /* Calculate per-expert memory */
    size_t expert_bytes =
        (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM * sizeof(float) +  /* W_in */
        (size_t)EXPERT_HIDDEN_DIM * sizeof(float) +                   /* b_in */
        (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM * sizeof(float) +  /* W_out */
        (size_t)EMBEDDING_DIM * sizeof(float);                        /* b_out */

    /* Dense equivalent */
    uint64_t dense_params = model->total_params;
    double dense_mb = (double)dense_params * sizeof(float) / (1024.0 * 1024.0);
    double active_pct = model->total_params > 0 ?
        100.0 * (double)model->active_params_last / (double)model->total_params : 0;

    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                    Model İstatistikleri                       ║\n");
    printf("╠════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                              ║\n");
    printf("║  Ağaç Yapısı                                                 ║\n");
    printf("║  ├─ Toplam düğüm    : %6u                                  ║\n", model->total_nodes);
    printf("║  ├─ Yaprak uzman    : %6u                                  ║\n", model->leaf_nodes);
    printf("║  ├─ Dallanma (B)    : %6d                                  ║\n", MAX_CHILDREN);
    printf("║  └─ Derinlik        : %6d seviye                           ║\n", TREE_DEPTH);
    printf("║                                                              ║\n");
    printf("║  Parametreler                                                ║\n");
    printf("║  ├─ Toplam parametre: %'12lu                              ║\n", (unsigned long)model->total_params);
    printf("║  ├─ Aktif (son)     : %'12lu                              ║\n", (unsigned long)model->active_params_last);
    printf("║  └─ Aktivasyon oranı: %8.2f%%                              ║\n", active_pct);
    printf("║                                                              ║\n");
    printf("║  Bellek                                                      ║\n");
    printf("║  ├─ Arena kullanımı : %8.2f MB                             ║\n", total_mb);
    printf("║  ├─ Uzman boyutu    : %8.2f KB (× %u)                     ║\n",
           (double)expert_bytes / 1024.0, model->leaf_nodes);
    printf("║  └─ Dense eşdeğeri  : %8.2f MB                             ║\n", dense_mb);
    printf("║                                                              ║\n");
    printf("║  ══════════════════════════════════════                       ║\n");
    printf("║                   HT-LLM vs Dense LLM                       ║\n");
    printf("║  ══════════════════════════════════════                       ║\n");
    printf("║  Token başına RAM   : ~%.0f KB  vs  ~14 GB (Dense 7B)       ║\n",
           (double)expert_bytes / 1024.0);
    printf("║  Hesaplama          : O(log N)     vs  O(N)                  ║\n");
    printf("║  Aktivasyon         : %%%.1f          vs  %%100               ║\n", active_pct);
    printf("║                                                              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
}

/* ══════════════════════════════════════════════════════════════
 *  Destroy — single free via arena
 * ══════════════════════════════════════════════════════════════ */

void htllm_destroy(HTLLMModel* model) {
    if (!model) return;
    MemoryArena* arena = model->arena;
    /* Single free releases all model memory */
    arena_destroy(arena);
}
