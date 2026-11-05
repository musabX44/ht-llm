/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Training Module
 *  Next-token prediction, Cross-Entropy loss, SGD backprop.
 *
 *  Eğitim stratejisi (sparse-MoE'ye uygun):
 *   - Her adımda htllm_forward ile aynı forward pass tekrar yapılır,
 *     ancak bu sefer ara değerler (embedding, context, seçilen
 *     yapraklar, expert hidden aktivasyonları, merged_output) da
 *     saklanır.
 *   - Loss = CrossEntropy(softmax(logits), target_token)
 *   - Geriye doğru: logits -> output_head (W_proj,b_proj) ->
 *     merged_output -> her seçili expert'in (W_out,b_out,W_in,b_in) ->
 *     context vektörü üzerinden embedding satırına gradyan.
 *   - Routing (centroid) ağırlıkları ve ağaç yapısı SABİT tutulur;
 *     yalnızca "aktif" (seçilmiş) uzmanlar ve ilgili embedding satırı
 *     güncellenir — bu da MoE'nin seyrek güncelleme felsefesiyle
 *     tutarlıdır.
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* İleri bildirim: tanım dosyanın altında (JSONL Desteği bölümünde) */
static int jsonl_extract_field(const char* line, const char* field_name,
                               char* out_buf, size_t out_buf_size);

/* ══════════════════════════════════════════════════════════════
 *  Tek Eğitim Adımı: forward + backward + SGD update
 * ══════════════════════════════════════════════════════════════ */

float htllm_train_step(HTLLMModel* model, uint32_t input_token,
                       uint32_t target_token, float learning_rate) {
    if (!model) return 0.0f;
    if (input_token >= model->embeddings->vocab_size) input_token = 0;
    if (target_token >= VOCAB_SIZE) target_token = 0;

    /* ── FORWARD PASS (ara değerleri saklayarak) ── */

    /* 1. Embedding lookup */
    alignas(CACHE_LINE_BYTES) float embedding[EMBEDDING_DIM];
    embedding_lookup(model->embeddings, input_token, embedding);

    /* 2. Context güncelle (EMA) */
    context_update(&model->context, embedding);

    /* ctx_vector: expert girişleri ve routing bu vektör üzerinden yapılır */
    alignas(CACHE_LINE_BYTES) float ctx_vector[EMBEDDING_DIM];
    vec_copy(ctx_vector, model->context.vector, EMBEDDING_DIM);

    /* 3. Ağaç yönlendirme (routing ağırlıkları eğitim dışı / sabit) */
    TreeNode* selected_leaves[TOP_K_MAX * TOP_K_MAX];
    float     leaf_weights[TOP_K_MAX * TOP_K_MAX];
    int       num_leaves = 0;
    tree_route_full(model->root, ctx_vector, selected_leaves, leaf_weights, &num_leaves);

    if (num_leaves <= 0) {
        return 0.0f; /* Yönlendirilecek uzman yoksa bu adımı atla */
    }

    /* 4. Her seçili uzman için forward'ı manuel yürüt (ara aktivasyonları sakla) */
    float hidden_pre[TOP_K_MAX * TOP_K_MAX][EXPERT_HIDDEN_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
    float hidden_act[TOP_K_MAX * TOP_K_MAX][EXPERT_HIDDEN_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
    float expert_out[TOP_K_MAX * TOP_K_MAX][EMBEDDING_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));

    alignas(CACHE_LINE_BYTES) float merged_output[EMBEDDING_DIM];
    vec_zero(merged_output, EMBEDDING_DIM);

    for (int i = 0; i < num_leaves; i++) {
        const MicroExpert* ex = selected_leaves[i]->expert;

        /* hidden_pre = W_in · ctx + b_in   (W_in: [HIDDEN x EMBED]) */
        mat_vec_mul(ex->W_in, ctx_vector, ex->b_in,
                    hidden_pre[i], EXPERT_HIDDEN_DIM, EMBEDDING_DIM);

        /* ReLU */
        vec_relu(hidden_act[i], hidden_pre[i], EXPERT_HIDDEN_DIM);

        /* out = W_out · hidden_act + b_out (W_out: [EMBED x HIDDEN]) */
        mat_vec_mul(ex->W_out, hidden_act[i], ex->b_out,
                    expert_out[i], EMBEDDING_DIM, EXPERT_HIDDEN_DIM);

        /* Ağırlıklı toplam */
        vec_weighted_add(merged_output, expert_out[i], leaf_weights[i], EMBEDDING_DIM);
    }

    /* 5. Output head → logits */
    float logits[VOCAB_SIZE] __attribute__((aligned(CACHE_LINE_BYTES)));
    output_head_forward(model->output_head, merged_output, logits);

    /* 6. Softmax (loss hesaplaması için ayrı kopya üstünde) */
    float probs[VOCAB_SIZE] __attribute__((aligned(CACHE_LINE_BYTES)));
    memcpy(probs, logits, sizeof(float) * VOCAB_SIZE);
    vec_softmax(probs, VOCAB_SIZE, SOFTMAX_TEMPERATURE);

    float p_target = probs[target_token];
    if (p_target < 1e-8f) p_target = 1e-8f;
    float loss = -logf(p_target);

    /* ── BACKWARD PASS ── */

    /* dL/dlogits = probs - one_hot(target)  (softmax + cross-entropy türevi) */
    float dlogits[VOCAB_SIZE] __attribute__((aligned(CACHE_LINE_BYTES)));
    memcpy(dlogits, probs, sizeof(float) * VOCAB_SIZE);
    dlogits[target_token] -= 1.0f;

    /* dL/dmerged_output[d] = Σ_v dlogits[v] * W_proj[d*VOCAB_SIZE + v]
     * ve aynı anda W_proj, b_proj için gradyan uygula (SGD, in-place). */
    float d_merged[EMBEDDING_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
    vec_zero(d_merged, EMBEDDING_DIM);

    OutputHead* head = model->output_head;
    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
        float mo = merged_output[d];
        float grad_sum = 0.0f;
        float* row = &head->W_proj[d * VOCAB_SIZE];
        for (size_t v = 0; v < VOCAB_SIZE; v++) {
            grad_sum += dlogits[v] * row[v];
            /* dL/dW_proj[d,v] = dlogits[v] * merged_output[d] */
            row[v] -= learning_rate * dlogits[v] * mo;
        }
        d_merged[d] = grad_sum;
    }
    for (size_t v = 0; v < VOCAB_SIZE; v++) {
        head->b_proj[v] -= learning_rate * dlogits[v];
    }

    /* Her seçili uzmana geri yay: d_merged * leaf_weight[i] o uzmanın
     * çıkışına gelen gradyan. */
    float d_ctx_total[EMBEDDING_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
    vec_zero(d_ctx_total, EMBEDDING_DIM);

    for (int i = 0; i < num_leaves; i++) {
        MicroExpert* ex = selected_leaves[i]->expert;
        float w = leaf_weights[i];

        /* d_out[d] = d_merged[d] * w */
        float d_out[EMBEDDING_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
        for (size_t d = 0; d < EMBEDDING_DIM; d++) d_out[d] = d_merged[d] * w;

        /* out = W_out · hidden_act + b_out
         * dL/dW_out[d, h] = d_out[d] * hidden_act[h]
         * dL/db_out[d]    = d_out[d]
         * dL/dhidden_act[h] = Σ_d d_out[d] * W_out[d, h] */
        float d_hidden_act[EXPERT_HIDDEN_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
        vec_zero(d_hidden_act, EXPERT_HIDDEN_DIM);

        for (size_t d = 0; d < EMBEDDING_DIM; d++) {
            float go = d_out[d];
            float* row = &ex->W_out[d * EXPERT_HIDDEN_DIM];
            for (size_t h = 0; h < EXPERT_HIDDEN_DIM; h++) {
                d_hidden_act[h] += go * row[h];
                row[h] -= learning_rate * go * hidden_act[i][h];
            }
            ex->b_out[d] -= learning_rate * go;
        }

        /* ReLU türevi */
        float d_hidden_pre[EXPERT_HIDDEN_DIM] __attribute__((aligned(CACHE_LINE_BYTES)));
        for (size_t h = 0; h < EXPERT_HIDDEN_DIM; h++) {
            d_hidden_pre[h] = (hidden_pre[i][h] > 0.0f) ? d_hidden_act[h] : 0.0f;
        }

        /* hidden_pre = W_in · ctx + b_in
         * dL/dW_in[h, e] = d_hidden_pre[h] * ctx[e]
         * dL/db_in[h]    = d_hidden_pre[h]
         * dL/dctx[e]    += Σ_h d_hidden_pre[h] * W_in[h, e] */
        for (size_t h = 0; h < EXPERT_HIDDEN_DIM; h++) {
            float gh = d_hidden_pre[h];
            float* row = &ex->W_in[h * EMBEDDING_DIM];
            for (size_t e = 0; e < EMBEDDING_DIM; e++) {
                d_ctx_total[e] += gh * row[e];
                row[e] -= learning_rate * gh * ctx_vector[e];
            }
            ex->b_in[h] -= learning_rate * gh;
        }
    }

    /* Embedding satırına geri yay (context = EMA(embedding), yaklaşık
     * olarak (1-alpha) katsayısıyla mevcut embedding'e akar). */
    float alpha = model->context.alpha;
    float* emb_row = &model->embeddings->embeddings[(size_t)input_token * EMBEDDING_DIM];
    for (size_t e = 0; e < EMBEDDING_DIM; e++) {
        emb_row[e] -= learning_rate * (1.0f - alpha) * d_ctx_total[e];
    }

    return loss;
}

/* ══════════════════════════════════════════════════════════════
 *  Dizi Üzerinde Eğitim (Autoregressive Next-Token Prediction)
 * ══════════════════════════════════════════════════════════════ */

void htllm_train_sequence(HTLLMModel* model, const uint32_t* token_ids,
                          int n_tokens, const TrainConfig* config) {
    if (!model || !token_ids || n_tokens < 2 || !config) {
        printf("  [Eğitim] Geçersiz parametreler, eğitim atlandı.\n");
        return;
    }

    float lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;
    int   epochs = config->epochs > 0 ? config->epochs : 1;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║              Eğitim (Training) Başlıyor                ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Token sayısı   : %-6d                                 ║\n", n_tokens);
    printf("  ║  Öğrenim oranı  : %-8.5f                              ║\n", lr);
    printf("  ║  Epoch sayısı   : %-6d                                 ║\n", epochs);
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = get_time_ms();
        float total_loss = 0.0f;
        int   n_steps = 0;

        /* Her epoch başında context'i sıfırla (bağımsız dizi varsayımı) */
        context_reset(&model->context);

        for (int i = 0; i < n_tokens - 1; i++) {
            uint32_t input_tok  = token_ids[i];
            uint32_t target_tok = token_ids[i + 1];

            float loss = htllm_train_step(model, input_tok, target_tok, lr);
            total_loss += loss;
            n_steps++;

            if (config->verbose) {
                printf("    [epoch %d][adım %d/%d] token %u → %u  loss=%.4f\n",
                       epoch + 1, i + 1, n_tokens - 1, input_tok, target_tok, loss);
            }
        }

        double t1 = get_time_ms();
        float avg_loss = n_steps > 0 ? total_loss / (float)n_steps : 0.0f;

        printf("  [Epoch %d/%d] ortalama loss = %.4f  │  perplexity ≈ %.3f  │  süre: %.2f ms\n",
               epoch + 1, epochs, avg_loss, expf(avg_loss), t1 - t0);
    }

    printf("\n  ✓ Eğitim tamamlandı.\n");
}

/* ══════════════════════════════════════════════════════════════
 *  Dosyadan Dataset Eğitimi
 *
 *  Dosyayı satır satır okur, her satırı tokenize eder ve
 *  htllm_train_sequence ile eğitir. Tüm satırlardaki toplam
 *  loss'un ortalaması, her epoch sonunda raporlanır.
 * ══════════════════════════════════════════════════════════════ */

int htllm_train_from_file(HTLLMModel* model, const char* filepath,
                          const TrainConfig* config, int max_tokens_per_line) {
    if (!model || !filepath || !config) return 0;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "  [Eğitim] HATA: Dosya açılamadı: %s\n", filepath);
        return 0;
    }

    if (max_tokens_per_line <= 0) max_tokens_per_line = 64;

    /* Önce satır sayısını say (ilerleme raporu için) */
    int total_lines = 0;
    {
        char probe[4096];
        while (fgets(probe, sizeof(probe), f)) {
            /* Sadece boş olmayan satırları say */
            int has_content = 0;
            for (char* c = probe; *c; c++) {
                if (!isspace((unsigned char)*c)) { has_content = 1; break; }
            }
            if (has_content) total_lines++;
        }
        rewind(f);
    }

    if (total_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Dosyada eğitilecek satır bulunamadı.\n");
        fclose(f);
        return 0;
    }

    float lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;
    int   epochs = config->epochs > 0 ? config->epochs : 1;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║          Dataset'ten Eğitim (Training from File)       ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Dosya           : %-40s ║\n", filepath);
    printf("  ║  Satır sayısı     : %-6d                               ║\n", total_lines);
    printf("  ║  Öğrenim oranı   : %-8.5f                             ║\n", lr);
    printf("  ║  Epoch sayısı    : %-6d                                ║\n", epochs);
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    /* Tüm satırları belleğe oku ve önceden tokenize et (epoch'lar arası
     * tekrar tekrar dosya okuma/tokenize maliyetinden kaçınmak için) */
    uint32_t** line_tokens = (uint32_t**)malloc((size_t)total_lines * sizeof(uint32_t*));
    int* line_lengths = (int*)malloc((size_t)total_lines * sizeof(int));
    if (!line_tokens || !line_lengths) {
        fprintf(stderr, "  [Eğitim] HATA: Bellek ayrılamadı.\n");
        free(line_tokens); free(line_lengths);
        fclose(f);
        return 0;
    }

    char line_buf[4096];
    int line_idx = 0;
    int skipped_short = 0;

    while (fgets(line_buf, sizeof(line_buf), f) && line_idx < total_lines) {
        /* Boş satırları atla */
        int has_content = 0;
        for (char* c = line_buf; *c; c++) {
            if (!isspace((unsigned char)*c)) { has_content = 1; break; }
        }
        if (!has_content) continue;

        uint32_t* ids = (uint32_t*)malloc((size_t)max_tokens_per_line * sizeof(uint32_t));
        if (!ids) continue;

        int n_tok = tokenizer_encode(model->tokenizer, line_buf, ids, max_tokens_per_line);

        if (n_tok < 2) {
            free(ids);
            skipped_short++;
            continue;
        }

        line_tokens[line_idx] = ids;
        line_lengths[line_idx] = n_tok;
        line_idx++;
    }
    fclose(f);

    int n_valid_lines = line_idx;

    if (skipped_short > 0) {
        printf("  [Eğitim] Not: %d satır çok kısa olduğu için atlandı.\n", skipped_short);
    }

    if (n_valid_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Eğitilebilir (yeterli token içeren) satır yok.\n");
        free(line_tokens); free(line_lengths);
        return 0;
    }

    /* Epoch döngüsü: tüm satırları sırayla eğit */
    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = get_time_ms();
        float total_loss = 0.0f;
        int   total_steps = 0;

        for (int li = 0; li < n_valid_lines; li++) {
            context_reset(&model->context);

            uint32_t* ids = line_tokens[li];
            int n_tok = line_lengths[li];

            for (int i = 0; i < n_tok - 1; i++) {
                float loss = htllm_train_step(model, ids[i], ids[i + 1], lr);
                total_loss += loss;
                total_steps++;
            }

            if (config->verbose && (li % 50 == 0)) {
                printf("    [epoch %d] satır %d/%d işleniyor...\n",
                       epoch + 1, li + 1, n_valid_lines);
            }
        }

        double t1 = get_time_ms();
        float avg_loss = total_steps > 0 ? total_loss / (float)total_steps : 0.0f;

        printf("  [Epoch %d/%d] satır=%d  adım=%d  ort.loss=%.4f  perplexity≈%.3f  süre=%.1f ms\n",
               epoch + 1, epochs, n_valid_lines, total_steps, avg_loss, expf(avg_loss), t1 - t0);
    }

    /* Belleği serbest bırak */
    for (int li = 0; li < n_valid_lines; li++) {
        free(line_tokens[li]);
    }
    free(line_tokens);
    free(line_lengths);

    printf("\n  ✓ Dataset eğitimi tamamlandı (%d satır, %d epoch).\n", n_valid_lines, epochs);
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  Resumable Eğitim (Checkpoint destekli)
 *
 *  Aşağıdaki iki fonksiyon, htllm_train_from_file / _jsonl ile aynı
 *  dosya okuma + tokenize mantığını kullanır, ancak:
 *   - start_epoch'tan başlayarak epoch döngüsünü sürdürür (resume),
 *   - her epoch sonunda on_epoch_end callback'ini çağırır (checkpoint
 *     kaydı için main.c bu callback'i kullanır).
 * ══════════════════════════════════════════════════════════════ */

int htllm_train_from_file_resumable(HTLLMModel* model, const char* filepath,
                          const TrainConfig* config, int max_tokens_per_line,
                          int start_epoch, uint64_t start_global_step,
                          HtllmEpochCallback on_epoch_end, void* user_data) {
    if (!model || !filepath || !config) return 0;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "  [Eğitim] HATA: Dosya açılamadı: %s\n", filepath);
        return 0;
    }
    if (max_tokens_per_line <= 0) max_tokens_per_line = 64;

    int total_lines = 0;
    {
        char probe[4096];
        while (fgets(probe, sizeof(probe), f)) {
            int has_content = 0;
            for (char* c = probe; *c; c++) {
                if (!isspace((unsigned char)*c)) { has_content = 1; break; }
            }
            if (has_content) total_lines++;
        }
        rewind(f);
    }

    if (total_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Dosyada eğitilecek satır bulunamadı.\n");
        fclose(f);
        return 0;
    }

    float lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;
    int   target_epochs = config->epochs > 0 ? config->epochs : 1;
    if (start_epoch < 0) start_epoch = 0;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║     Dataset'ten Eğitim (Resumable, Training from File)  ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Dosya           : %-40s ║\n", filepath);
    printf("  ║  Satır sayısı     : %-6d                               ║\n", total_lines);
    printf("  ║  Öğrenim oranı   : %-8.5f                             ║\n", lr);
    printf("  ║  Hedef epoch     : %-6d  (başlangıç: %-6d)          ║\n", target_epochs, start_epoch);
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    if (start_epoch >= target_epochs) {
        printf("  [Eğitim] start_epoch (%d) >= hedef epoch (%d); yapılacak iş yok.\n",
               start_epoch, target_epochs);
        fclose(f);
        return 1;
    }

    uint32_t** line_tokens = (uint32_t**)malloc((size_t)total_lines * sizeof(uint32_t*));
    int* line_lengths = (int*)malloc((size_t)total_lines * sizeof(int));
    if (!line_tokens || !line_lengths) {
        fprintf(stderr, "  [Eğitim] HATA: Bellek ayrılamadı.\n");
        free(line_tokens); free(line_lengths);
        fclose(f);
        return 0;
    }

    char line_buf[4096];
    int line_idx = 0;
    int skipped_short = 0;

    while (fgets(line_buf, sizeof(line_buf), f) && line_idx < total_lines) {
        int has_content = 0;
        for (char* c = line_buf; *c; c++) {
            if (!isspace((unsigned char)*c)) { has_content = 1; break; }
        }
        if (!has_content) continue;

        uint32_t* ids = (uint32_t*)malloc((size_t)max_tokens_per_line * sizeof(uint32_t));
        if (!ids) continue;

        int n_tok = tokenizer_encode(model->tokenizer, line_buf, ids, max_tokens_per_line);
        if (n_tok < 2) {
            free(ids);
            skipped_short++;
            continue;
        }

        line_tokens[line_idx] = ids;
        line_lengths[line_idx] = n_tok;
        line_idx++;
    }
    fclose(f);

    int n_valid_lines = line_idx;
    if (skipped_short > 0) {
        printf("  [Eğitim] Not: %d satır çok kısa olduğu için atlandı.\n", skipped_short);
    }
    if (n_valid_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Eğitilebilir (yeterli token içeren) satır yok.\n");
        free(line_tokens); free(line_lengths);
        return 0;
    }

    uint64_t global_step = start_global_step;

    for (int epoch = start_epoch; epoch < target_epochs; epoch++) {
        double t0 = get_time_ms();
        float total_loss = 0.0f;
        int   total_steps = 0;

        for (int li = 0; li < n_valid_lines; li++) {
            context_reset(&model->context);
            uint32_t* ids = line_tokens[li];
            int n_tok = line_lengths[li];

            for (int i = 0; i < n_tok - 1; i++) {
                float loss = htllm_train_step(model, ids[i], ids[i + 1], lr);
                total_loss += loss;
                total_steps++;
                global_step++;
            }

            if (config->verbose && (li % 50 == 0)) {
                printf("    [epoch %d] satır %d/%d işleniyor...\n",
                       epoch + 1, li + 1, n_valid_lines);
            }
        }

        double t1 = get_time_ms();
        float avg_loss = total_steps > 0 ? total_loss / (float)total_steps : 0.0f;

        printf("  [Epoch %d/%d] satır=%d  adım=%d  ort.loss=%.4f  perplexity≈%.3f  süre=%.1f ms\n",
               epoch + 1, target_epochs, n_valid_lines, total_steps, avg_loss, expf(avg_loss), t1 - t0);

        if (on_epoch_end) {
            on_epoch_end(epoch + 1, avg_loss, global_step, user_data);
        }
    }

    for (int li = 0; li < n_valid_lines; li++) free(line_tokens[li]);
    free(line_tokens);
    free(line_lengths);

    printf("\n  ✓ Dataset eğitimi tamamlandı (%d satır, %d→%d epoch).\n",
           n_valid_lines, start_epoch, target_epochs);
    return 1;
}

int htllm_train_from_jsonl_resumable(HTLLMModel* model, const char* filepath,
                           const TrainConfig* config, int max_tokens_per_line,
                           const char* field_name,
                           int start_epoch, uint64_t start_global_step,
                           HtllmEpochCallback on_epoch_end, void* user_data) {
    if (!model || !filepath || !config) return 0;
    if (!field_name) field_name = "text";
    if (max_tokens_per_line <= 0) max_tokens_per_line = 64;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "  [Eğitim] HATA: Dosya açılamadı: %s\n", filepath);
        return 0;
    }

    int total_lines = 0;
    {
        char probe[8192];
        while (fgets(probe, sizeof(probe), f)) {
            int has_content = 0;
            for (char* c = probe; *c; c++) {
                if (!isspace((unsigned char)*c)) { has_content = 1; break; }
            }
            if (has_content) total_lines++;
        }
        rewind(f);
    }

    if (total_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Dosyada eğitilecek satır bulunamadı.\n");
        fclose(f);
        return 0;
    }

    float lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;
    int   target_epochs = config->epochs > 0 ? config->epochs : 1;
    if (start_epoch < 0) start_epoch = 0;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║   JSONL Dataset'ten Eğitim (Resumable, Training)         ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Dosya           : %-40s ║\n", filepath);
    printf("  ║  JSON alanı      : %-40s ║\n", field_name);
    printf("  ║  Satır sayısı     : %-6d                               ║\n", total_lines);
    printf("  ║  Öğrenim oranı   : %-8.5f                             ║\n", lr);
    printf("  ║  Hedef epoch     : %-6d  (başlangıç: %-6d)          ║\n", target_epochs, start_epoch);
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    if (start_epoch >= target_epochs) {
        printf("  [Eğitim] start_epoch (%d) >= hedef epoch (%d); yapılacak iş yok.\n",
               start_epoch, target_epochs);
        fclose(f);
        return 1;
    }

    uint32_t** line_tokens = (uint32_t**)malloc((size_t)total_lines * sizeof(uint32_t*));
    int* line_lengths = (int*)malloc((size_t)total_lines * sizeof(int));
    if (!line_tokens || !line_lengths) {
        fprintf(stderr, "  [Eğitim] HATA: Bellek ayrılamadı.\n");
        free(line_tokens); free(line_lengths);
        fclose(f);
        return 0;
    }

    char raw_line[8192];
    char text_buf[4096];
    int line_idx = 0;
    int skipped_short = 0;
    int skipped_no_field = 0;
    int raw_line_no = 0;

    while (fgets(raw_line, sizeof(raw_line), f) && line_idx < total_lines) {
        raw_line_no++;
        int has_content = 0;
        for (char* c = raw_line; *c; c++) {
            if (!isspace((unsigned char)*c)) { has_content = 1; break; }
        }
        if (!has_content) continue;

        if (!jsonl_extract_field(raw_line, field_name, text_buf, sizeof(text_buf))) {
            skipped_no_field++;
            if (config->verbose) {
                fprintf(stderr, "  [Eğitim] Uyarı: satır %d içinde \"%s\" alanı bulunamadı, atlanıyor.\n",
                        raw_line_no, field_name);
            }
            continue;
        }

        uint32_t* ids = (uint32_t*)malloc((size_t)max_tokens_per_line * sizeof(uint32_t));
        if (!ids) continue;

        int n_tok = tokenizer_encode(model->tokenizer, text_buf, ids, max_tokens_per_line);
        if (n_tok < 2) {
            free(ids);
            skipped_short++;
            continue;
        }

        line_tokens[line_idx] = ids;
        line_lengths[line_idx] = n_tok;
        line_idx++;
    }
    fclose(f);

    int n_valid_lines = line_idx;
    if (skipped_no_field > 0) {
        printf("  [Eğitim] Not: %d satırda \"%s\" alanı bulunamadığı için atlandı.\n",
               skipped_no_field, field_name);
    }
    if (skipped_short > 0) {
        printf("  [Eğitim] Not: %d satır çok kısa olduğu için atlandı.\n", skipped_short);
    }
    if (n_valid_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Eğitilebilir satır yok. \"%s\" alanını ve JSON formatını kontrol edin.\n",
                field_name);
        free(line_tokens); free(line_lengths);
        return 0;
    }

    uint64_t global_step = start_global_step;

    for (int epoch = start_epoch; epoch < target_epochs; epoch++) {
        double t0 = get_time_ms();
        float total_loss = 0.0f;
        int   total_steps = 0;

        for (int li = 0; li < n_valid_lines; li++) {
            context_reset(&model->context);
            uint32_t* ids = line_tokens[li];
            int n_tok = line_lengths[li];

            for (int i = 0; i < n_tok - 1; i++) {
                float loss = htllm_train_step(model, ids[i], ids[i + 1], lr);
                total_loss += loss;
                total_steps++;
                global_step++;
            }

            if (config->verbose && (li % 50 == 0)) {
                printf("    [epoch %d] kayıt %d/%d işleniyor...\n",
                       epoch + 1, li + 1, n_valid_lines);
            }
        }

        double t1 = get_time_ms();
        float avg_loss = total_steps > 0 ? total_loss / (float)total_steps : 0.0f;

        printf("  [Epoch %d/%d] kayıt=%d  adım=%d  ort.loss=%.4f  perplexity≈%.3f  süre=%.1f ms\n",
               epoch + 1, target_epochs, n_valid_lines, total_steps, avg_loss, expf(avg_loss), t1 - t0);

        if (on_epoch_end) {
            on_epoch_end(epoch + 1, avg_loss, global_step, user_data);
        }
    }

    for (int li = 0; li < n_valid_lines; li++) free(line_tokens[li]);
    free(line_tokens);
    free(line_lengths);

    printf("\n  ✓ JSONL dataset eğitimi tamamlandı (%d kayıt, %d→%d epoch).\n",
           n_valid_lines, start_epoch, target_epochs);
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  JSONL Desteği
 *
 *  Basit, minimal bir JSON string-değer çıkarıcı. Tam bir JSON
 *  parser değildir — yalnızca düz {"field": "değer", ...} yapılı,
 *  kaçış (\") karakteri içermeyen tek satırlık nesneleri hedefler.
 *  Amaç: "text" gibi bir alandaki metni bulup tokenize etmek.
 * ══════════════════════════════════════════════════════════════ */

/* line içinde "field_name": "..." kalıbını arar, değeri out_buf'a
 * kopyalar (\uXXXX ve \n \t \" \\ kaçışlarını çözer). Bulunamazsa 0
 * döner. */
static int jsonl_extract_field(const char* line, const char* field_name,
                               char* out_buf, size_t out_buf_size) {
    if (!line || !field_name || !out_buf || out_buf_size == 0) return 0;

    /* "field_name" dizisini ara */
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", field_name);

    const char* key_pos = strstr(line, needle);
    if (!key_pos) return 0;

    /* Anahtardan sonra ':' bul */
    const char* p = key_pos + strlen(needle);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return 0;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;

    /* Değer bir string mi ("...")? Şimdilik yalnızca string alanları
     * destekleniyor. */
    if (*p != '"') return 0;
    p++;

    size_t out_i = 0;
    while (*p && *p != '"' && out_i + 1 < out_buf_size) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case 'n': out_buf[out_i++] = '\n'; break;
                case 't': out_buf[out_i++] = '\t'; break;
                case 'r': out_buf[out_i++] = '\r'; break;
                case '"': out_buf[out_i++] = '"';  break;
                case '\\': out_buf[out_i++] = '\\'; break;
                case '/': out_buf[out_i++] = '/';  break;
                case 'u': {
                    /* \uXXXX: sadece ASCII aralığındaki (0000-00FF)
                     * kod noktalarını tek byte olarak kabaca çözer;
                     * bu prototip için yeterlidir, tam UTF-8 değildir. */
                    if (isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2]) &&
                        isxdigit((unsigned char)p[3]) && isxdigit((unsigned char)p[4])) {
                        char hex[5] = { p[1], p[2], p[3], p[4], 0 };
                        long code = strtol(hex, NULL, 16);
                        if (code > 0 && code < 256) out_buf[out_i++] = (char)code;
                        p += 4;
                    }
                    break;
                }
                default: out_buf[out_i++] = *p; break;
            }
            p++;
        } else {
            out_buf[out_i++] = *p++;
        }
    }
    out_buf[out_i] = '\0';

    return out_i > 0;
}

int htllm_train_from_jsonl(HTLLMModel* model, const char* filepath,
                           const TrainConfig* config, int max_tokens_per_line,
                           const char* field_name) {
    if (!model || !filepath || !config) return 0;
    if (!field_name) field_name = "text";
    if (max_tokens_per_line <= 0) max_tokens_per_line = 64;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "  [Eğitim] HATA: Dosya açılamadı: %s\n", filepath);
        return 0;
    }

    /* Satır sayısını say */
    int total_lines = 0;
    {
        char probe[8192];
        while (fgets(probe, sizeof(probe), f)) {
            int has_content = 0;
            for (char* c = probe; *c; c++) {
                if (!isspace((unsigned char)*c)) { has_content = 1; break; }
            }
            if (has_content) total_lines++;
        }
        rewind(f);
    }

    if (total_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Dosyada eğitilecek satır bulunamadı.\n");
        fclose(f);
        return 0;
    }

    float lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;
    int   epochs = config->epochs > 0 ? config->epochs : 1;

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║        JSONL Dataset'ten Eğitim (Training from JSONL)   ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Dosya           : %-40s ║\n", filepath);
    printf("  ║  JSON alanı      : %-40s ║\n", field_name);
    printf("  ║  Satır sayısı     : %-6d                               ║\n", total_lines);
    printf("  ║  Öğrenim oranı   : %-8.5f                             ║\n", lr);
    printf("  ║  Epoch sayısı    : %-6d                                ║\n", epochs);
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    uint32_t** line_tokens = (uint32_t**)malloc((size_t)total_lines * sizeof(uint32_t*));
    int* line_lengths = (int*)malloc((size_t)total_lines * sizeof(int));
    if (!line_tokens || !line_lengths) {
        fprintf(stderr, "  [Eğitim] HATA: Bellek ayrılamadı.\n");
        free(line_tokens); free(line_lengths);
        fclose(f);
        return 0;
    }

    char raw_line[8192];
    char text_buf[4096];
    int line_idx = 0;
    int skipped_short = 0;
    int skipped_no_field = 0;
    int raw_line_no = 0;

    while (fgets(raw_line, sizeof(raw_line), f) && line_idx < total_lines) {
        raw_line_no++;

        int has_content = 0;
        for (char* c = raw_line; *c; c++) {
            if (!isspace((unsigned char)*c)) { has_content = 1; break; }
        }
        if (!has_content) continue;

        if (!jsonl_extract_field(raw_line, field_name, text_buf, sizeof(text_buf))) {
            skipped_no_field++;
            if (config->verbose) {
                fprintf(stderr, "  [Eğitim] Uyarı: satır %d içinde \"%s\" alanı bulunamadı, atlanıyor.\n",
                        raw_line_no, field_name);
            }
            continue;
        }

        uint32_t* ids = (uint32_t*)malloc((size_t)max_tokens_per_line * sizeof(uint32_t));
        if (!ids) continue;

        int n_tok = tokenizer_encode(model->tokenizer, text_buf, ids, max_tokens_per_line);

        if (n_tok < 2) {
            free(ids);
            skipped_short++;
            continue;
        }

        line_tokens[line_idx] = ids;
        line_lengths[line_idx] = n_tok;
        line_idx++;
    }
    fclose(f);

    int n_valid_lines = line_idx;

    if (skipped_no_field > 0) {
        printf("  [Eğitim] Not: %d satırda \"%s\" alanı bulunamadığı için atlandı.\n",
               skipped_no_field, field_name);
    }
    if (skipped_short > 0) {
        printf("  [Eğitim] Not: %d satır çok kısa olduğu için atlandı.\n", skipped_short);
    }

    if (n_valid_lines == 0) {
        fprintf(stderr, "  [Eğitim] HATA: Eğitilebilir satır yok. \"%s\" alanını ve JSON formatını kontrol edin.\n",
                field_name);
        free(line_tokens); free(line_lengths);
        return 0;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = get_time_ms();
        float total_loss = 0.0f;
        int   total_steps = 0;

        for (int li = 0; li < n_valid_lines; li++) {
            context_reset(&model->context);

            uint32_t* ids = line_tokens[li];
            int n_tok = line_lengths[li];

            for (int i = 0; i < n_tok - 1; i++) {
                float loss = htllm_train_step(model, ids[i], ids[i + 1], lr);
                total_loss += loss;
                total_steps++;
            }

            if (config->verbose && (li % 50 == 0)) {
                printf("    [epoch %d] kayıt %d/%d işleniyor...\n",
                       epoch + 1, li + 1, n_valid_lines);
            }
        }

        double t1 = get_time_ms();
        float avg_loss = total_steps > 0 ? total_loss / (float)total_steps : 0.0f;

        printf("  [Epoch %d/%d] kayıt=%d  adım=%d  ort.loss=%.4f  perplexity≈%.3f  süre=%.1f ms\n",
               epoch + 1, epochs, n_valid_lines, total_steps, avg_loss, expf(avg_loss), t1 - t0);
    }

    for (int li = 0; li < n_valid_lines; li++) {
        free(line_tokens[li]);
    }
    free(line_tokens);
    free(line_lengths);

    printf("\n  ✓ JSONL dataset eğitimi tamamlandı (%d kayıt, %d epoch).\n", n_valid_lines, epochs);
    return 1;
}
