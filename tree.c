/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Hierarchical Tree — Construction & Routing Engine
 *
 *  3-Level Semantic Tree:
 *    Level 0: Root (1 node)
 *    Level 1: Domain Centroids (8 nodes)
 *    Level 2: Sub-Domain Centroids (64 nodes)
 *    Level 3: Leaf Micro-Experts (512 nodes)
 *
 *  Routing: O(log_B N) — dot product → softmax → top-k → prefetch
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ── Semantic domain labels for visualization ── */
static const char* DOMAIN_NAMES[MAX_CHILDREN] = {
    "Bilim/Fen",      "Teknoloji",     "Sosyal/İnsan",   "Sanat/Edebiyat",
    "Günlük Yaşam",   "Matematik",     "Sağlık/Tıp",     "Doğa/Çevre"
};

static const char* SUBDOMAIN_TEMPLATE[MAX_CHILDREN] = {
    "Temel",   "Uygulamalı",  "Teorik",    "Deneysel",
    "Tarihsel","Karşılaştırmalı", "Modern", "İleri"
};

/* ══════════════════════════════════════════════════════════════
 *  Tree Node Create — allocate from arena (cache-line aligned)
 * ══════════════════════════════════════════════════════════════ */

TreeNode* tree_create_node(MemoryArena* arena, uint32_t node_id, uint32_t depth) {
    TreeNode* node = (TreeNode*)arena_alloc(arena, sizeof(TreeNode));
    if (!node) return NULL;

    memset(node, 0, sizeof(TreeNode));
    node->node_id     = node_id;
    node->depth       = depth;
    node->child_count = 0;
    node->is_leaf     = 0;
    node->expert      = NULL;

    for (int i = 0; i < MAX_CHILDREN; i++) {
        node->children[i] = NULL;
    }

    return node;
}

/* ══════════════════════════════════════════════════════════════
 *  Generate semi-orthogonal centroid vectors
 *  Each level's centroids are spread across the embedding space
 *  to maximize routing discrimination.
 * ══════════════════════════════════════════════════════════════ */

static void generate_centroids(float* centroid, uint32_t index, uint32_t level,
                                uint32_t seed) {
    uint32_t rng_state[4];
    xoshiro128p_seed(rng_state, seed ^ (index * 31337) ^ (level * 7919));

    /* Generate random direction */
    for (int d = 0; d < EMBEDDING_DIM; d++) {
        centroid[d] = random_normal(rng_state);
    }

    /* Add structured bias based on index to spread centroids apart */
    int block_size = EMBEDDING_DIM / MAX_CHILDREN;
    int start = (int)(index % MAX_CHILDREN) * block_size;
    for (int d = start; d < start + block_size && d < EMBEDDING_DIM; d++) {
        centroid[d] += 2.0f;  /* Bias towards this subspace */
    }

    /* L2 normalize to unit sphere */
    vec_l2_normalize(centroid, EMBEDDING_DIM);
}

/* ══════════════════════════════════════════════════════════════
 *  Build Random 3-Level Tree
 *
 *  Structure:
 *    Root (L0) → 8 Domain Centroids (L1) → 64 Sub-Domain (L2) → 512 Leaves (L3)
 *
 *  All allocated from a single contiguous arena.
 * ══════════════════════════════════════════════════════════════ */

void tree_build_random(MemoryArena* arena, TreeNode* root, uint32_t seed) {
    if (!arena || !root) return;

    uint32_t node_counter = 1;
    uint32_t expert_counter = 0;

    printf("  [Ağaç] İnşa ediliyor: %d seviye, B=%d dallanma faktörü...\n",
           TREE_DEPTH, MAX_CHILDREN);

    /* Generate root centroid (covers entire space) */
    for (int d = 0; d < EMBEDDING_DIM; d++) {
        root->centroid[d] = 1.0f / sqrtf((float)EMBEDDING_DIM);
    }

    /* ── Level 1: Domain Centroids ── */
    root->child_count = MAX_CHILDREN;
    for (int i = 0; i < MAX_CHILDREN; i++) {
        TreeNode* l1_node = tree_create_node(arena, node_counter++, 1);
        if (!l1_node) { fprintf(stderr, "L1 node alloc failed\n"); return; }

        generate_centroids(l1_node->centroid, (uint32_t)i, 1, seed);
        root->children[i] = l1_node;

        /* ── Level 2: Sub-Domain Centroids ── */
        l1_node->child_count = MAX_CHILDREN;
        for (int j = 0; j < MAX_CHILDREN; j++) {
            TreeNode* l2_node = tree_create_node(arena, node_counter++, 2);
            if (!l2_node) { fprintf(stderr, "L2 node alloc failed\n"); return; }

            generate_centroids(l2_node->centroid, (uint32_t)(i * MAX_CHILDREN + j), 2,
                             seed + 1000);
            l1_node->children[j] = l2_node;

            /* ── Level 3: Leaf Nodes with Micro-Experts ── */
            l2_node->child_count = MAX_CHILDREN;
            for (int k = 0; k < MAX_CHILDREN; k++) {
                TreeNode* leaf = tree_create_node(arena, node_counter++, 3);
                if (!leaf) { fprintf(stderr, "Leaf node alloc failed\n"); return; }

                generate_centroids(leaf->centroid,
                    (uint32_t)(i * MAX_CHILDREN * MAX_CHILDREN + j * MAX_CHILDREN + k),
                    3, seed + 2000);
                leaf->is_leaf = 1;

                /* Create and initialize Micro-Expert */
                leaf->expert = expert_create(arena, expert_counter);
                if (leaf->expert) {
                    expert_init_random(leaf->expert, seed + expert_counter);
                    expert_counter++;
                }

                l2_node->children[k] = leaf;
            }
        }
    }

    printf("  [Ağaç] Tamamlandı: %u düğüm, %u yaprak uzman oluşturuldu.\n",
           node_counter, expert_counter);
}

/* ══════════════════════════════════════════════════════════════
 *  Route Step — single-level routing with entropy-adaptive k
 *
 *  1. Compute dot product similarity for all children
 *  2. Apply temperature-scaled softmax
 *  3. Calculate entropy H(P)
 *  4. Dynamically select k based on entropy:
 *     - H < 0.5  → k=1 (model is confident)
 *     - H > 1.8  → k=4 (model is uncertain, explore more)
 *     - else     → k=2 (default)
 *  5. Prune branches below θ = 0.05
 *  6. Prefetch next-level nodes
 * ══════════════════════════════════════════════════════════════ */

void tree_route_step(TreeNode* node, const float* ctx_vector,
                     float temperature, RoutingResult* result) {
    if (!node || !ctx_vector || !result) return;

    uint32_t n_children = node->child_count;
    if (n_children == 0) return;

    /* Step 1: Compute similarities via dot product */
    float similarities[MAX_CHILDREN];
    for (uint32_t i = 0; i < n_children; i++) {
        if (node->children[i]) {
            similarities[i] = vec_dot_product(ctx_vector,
                                              node->children[i]->centroid,
                                              EMBEDDING_DIM);
        } else {
            similarities[i] = -1e9f;
        }
    }

    /* Step 2: Temperature-scaled softmax */
    float probs[MAX_CHILDREN];
    for (uint32_t i = 0; i < n_children; i++) {
        probs[i] = similarities[i];
    }
    vec_softmax(probs, n_children, temperature);

    /* Step 3: Compute entropy */
    float entropy = vec_entropy(probs, n_children);
    result->entropy = entropy;

    /* Step 4: Dynamic k selection based on entropy */
    int k;
    if (entropy < ENTROPY_LOW_THRESH) {
        k = 1;   /* Model emin → tek dal */
    } else if (entropy > ENTROPY_HIGH_THRESH) {
        k = TOP_K_MAX;   /* Model belirsiz → geniş arama */
    } else {
        k = TOP_K_DEFAULT;  /* Normal → 2 dal */
    }

    /* Step 5: Top-k selection with pruning */
    int top_indices[TOP_K_MAX];
    float top_values[TOP_K_MAX];
    int actual_k = vec_top_k(probs, n_children, k, top_indices, top_values);

    /* Apply pruning threshold */
    int final_k = 0;
    for (int i = 0; i < actual_k; i++) {
        if (top_values[i] >= PRUNING_THRESHOLD) {
            result->selected_indices[final_k] = top_indices[i];
            result->selected_probs[final_k]   = top_values[i];
            final_k++;
        }
    }

    /* Ensure at least 1 branch is selected */
    if (final_k == 0 && actual_k > 0) {
        result->selected_indices[0] = top_indices[0];
        result->selected_probs[0]   = top_values[0];
        final_k = 1;
    }

    result->k = final_k;
    result->max_similarity = final_k > 0 ? similarities[result->selected_indices[0]] : 0.0f;

    /* Renormalize selected probabilities */
    float prob_sum = 0.0f;
    for (int i = 0; i < final_k; i++) {
        prob_sum += result->selected_probs[i];
    }
    if (prob_sum > 0.0f) {
        for (int i = 0; i < final_k; i++) {
            result->selected_probs[i] /= prob_sum;
        }
    }

    /* Step 6: Prefetch next-level nodes into L2 cache */
#ifdef __AVX2__
    for (int i = 0; i < final_k; i++) {
        int idx = result->selected_indices[i];
        TreeNode* next = node->children[idx];
        if (next) {
            _mm_prefetch((const char*)next, _MM_HINT_T0);
            _mm_prefetch((const char*)next->centroid, _MM_HINT_T0);
            if (next->expert) {
                _mm_prefetch((const char*)next->expert, _MM_HINT_T1);
                if (next->expert->W_in) {
                    _mm_prefetch((const char*)next->expert->W_in, _MM_HINT_T1);
                }
            }
        }
    }
#endif
}

/* ══════════════════════════════════════════════════════════════
 *  Full Tree Routing — Root to Leaf(s)
 *
 *  Traverses 3 levels with top-k branching.
 *  Returns selected leaf nodes and their routing weights.
 *
 *  Complexity: O(B · Depth) = O(8 × 3) = O(24) comparisons
 *              vs O(N) for dense models
 * ══════════════════════════════════════════════════════════════ */

void tree_route_full(TreeNode* root, const float* ctx_vector,
                     TreeNode** selected_leaves, float* leaf_weights,
                     int* num_leaves) {
    if (!root || !ctx_vector || !selected_leaves || !leaf_weights || !num_leaves) return;

    *num_leaves = 0;

    /* Use arrays as a BFS-like frontier */
    TreeNode* frontier[TOP_K_MAX * TOP_K_MAX * TOP_K_MAX];
    float     frontier_weights[TOP_K_MAX * TOP_K_MAX * TOP_K_MAX];
    int       frontier_count = 0;

    /* Start at root */
    TreeNode* current[TOP_K_MAX * TOP_K_MAX * TOP_K_MAX];
    float     current_weights[TOP_K_MAX * TOP_K_MAX * TOP_K_MAX];
    int       current_count = 1;
    current[0] = root;
    current_weights[0] = 1.0f;

    /* Traverse each level */
    for (int level = 0; level < TREE_DEPTH; level++) {
        frontier_count = 0;

        for (int c = 0; c < current_count; c++) {
            TreeNode* node = current[c];
            float     base_weight = current_weights[c];

            if (!node || node->child_count == 0) continue;

            /* Route at this node */
            RoutingResult result;
            memset(&result, 0, sizeof(result));
            tree_route_step(node, ctx_vector, SOFTMAX_TEMPERATURE, &result);

            /* Add selected children to next frontier */
            for (int i = 0; i < result.k; i++) {
                int idx = result.selected_indices[i];
                if (idx >= 0 && idx < (int)node->child_count && node->children[idx]) {
                    if (frontier_count < TOP_K_MAX * TOP_K_MAX * TOP_K_MAX) {
                        frontier[frontier_count]         = node->children[idx];
                        frontier_weights[frontier_count] = base_weight * result.selected_probs[i];
                        frontier_count++;
                    }
                }
            }
        }

        /* Move frontier to current */
        current_count = frontier_count;
        for (int i = 0; i < frontier_count; i++) {
            current[i] = frontier[i];
            current_weights[i] = frontier_weights[i];
        }
    }

    /* Collect leaf nodes */
    int leaf_count = 0;
    for (int i = 0; i < current_count && leaf_count < TOP_K_MAX * TOP_K_MAX; i++) {
        if (current[i] && current[i]->is_leaf && current[i]->expert) {
            selected_leaves[leaf_count] = current[i];
            leaf_weights[leaf_count]    = current_weights[i];
            leaf_count++;
        }
    }

    *num_leaves = leaf_count;
}

/* ══════════════════════════════════════════════════════════════
 *  Print Routing Path — visualization for debugging
 * ══════════════════════════════════════════════════════════════ */

void tree_print_path(TreeNode* root, const float* ctx_vector) {
    if (!root || !ctx_vector) return;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║           Ağaç Yönlendirme Yolu (Routing Path)         ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");

    TreeNode* node = root;
    int level = 0;

    while (node && !node->is_leaf && node->child_count > 0) {
        RoutingResult result;
        memset(&result, 0, sizeof(result));
        tree_route_step(node, ctx_vector, SOFTMAX_TEMPERATURE, &result);

        /* Print level info */
        printf("  ║ Seviye %d", level);
        if (level == 0)      printf(" (Kök)         ");
        else if (level == 1) printf(" (Alan)        ");
        else                 printf(" (Alt-Alan)    ");

        printf("│ k=%d, H=%.3f", result.k, result.entropy);
        if (result.entropy < ENTROPY_LOW_THRESH)      printf(" [EMİN]    ");
        else if (result.entropy > ENTROPY_HIGH_THRESH) printf(" [BELİRSİZ]");
        else                                           printf(" [NORMAL]  ");
        printf("  ║\n");

        for (int i = 0; i < result.k; i++) {
            int idx = result.selected_indices[i];
            printf("  ║   ");
            if (i == 0) printf("├─▶ ");
            else        printf("├── ");

            if (level == 0 && idx < MAX_CHILDREN) {
                printf("Dal[%d] %s", idx, DOMAIN_NAMES[idx]);
            } else if (level == 1 && idx < MAX_CHILDREN) {
                printf("Dal[%d] %s", idx, SUBDOMAIN_TEMPLATE[idx]);
            } else {
                printf("Dal[%d]", idx);
            }
            printf(" (P=%.3f, sim=%.3f)", result.selected_probs[i],
                   vec_dot_product(ctx_vector,
                                  node->children[idx]->centroid,
                                  EMBEDDING_DIM));
            if (i == 0) printf(" ★");
            printf("    ║\n");
        }

        /* Follow best path for visualization */
        int best_idx = result.selected_indices[0];
        node = node->children[best_idx];
        level++;
    }

    if (node && node->is_leaf) {
        printf("  ║ Seviye %d (Yaprak)      │ Uzman #%u, %zu parametre",
               level, node->expert ? node->expert->expert_id : 0,
               node->expert ? expert_param_count(node->expert) : 0);
        printf("  ║\n");
    }

    printf("  ╚══════════════════════════════════════════════════════════╝\n");
}

/* ══════════════════════════════════════════════════════════════
 *  Utility: Count nodes and leaves
 * ══════════════════════════════════════════════════════════════ */

uint32_t tree_count_nodes(const TreeNode* root) {
    if (!root) return 0;
    uint32_t count = 1;
    for (uint32_t i = 0; i < root->child_count; i++) {
        count += tree_count_nodes(root->children[i]);
    }
    return count;
}

uint32_t tree_count_leaves(const TreeNode* root) {
    if (!root) return 0;
    if (root->is_leaf) return 1;
    uint32_t count = 0;
    for (uint32_t i = 0; i < root->child_count; i++) {
        count += tree_count_leaves(root->children[i]);
    }
    return count;
}
