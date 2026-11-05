/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Demo & Benchmark Program
 *
 *  Demonstrasyonlar:
 *  1. Model inşası ve bellek istatistikleri
 *  2. Yönlendirme yolu görselleştirmesi
 *  3. Bağlam takibi ve topic drift tespiti
 *  4. Token üretimi ve latency benchmark
 *  5. Dense vs Sparse karşılaştırma tablosu
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════
 *  ANSI Color Codes (terminal çıktısı güzelleştirme)
 * ══════════════════════════════════════════════════════════════ */

#define CLR_RESET   "\033[0m"
#define CLR_BOLD    "\033[1m"
#define CLR_RED     "\033[31m"
#define CLR_GREEN   "\033[32m"
#define CLR_YELLOW  "\033[33m"
#define CLR_BLUE    "\033[34m"
#define CLR_MAGENTA "\033[35m"
#define CLR_CYAN    "\033[36m"
#define CLR_DIM     "\033[2m"

/* ══════════════════════════════════════════════════════════════
 *  Chat Modu — İnteraktif REPL
 *  Kullanıcıdan satır satır girdi okur, tokenize eder, modeli
 *  prompt üzerinden ileri besler ve otoregresif olarak
 *  (greedy argmax) yanıt üretir. Her tur öncesi bağlam (context)
 *  sıfırlanır, böylece turlar birbirine karışmaz.
 *
 *  Not: Bu ağırlıklar eğitilmemiş/az eğitilmiş olabilir (rastgele
 *  init + varsa --resume ile yüklenen checkpoint). Model, sabit
 *  128 kelimelik prototip kelime haznesinin dışındaki her kelimeyi
 *  <unk> olarak kodlar; bu yüzden yanıtlar tutarsız görünebilir.
 *  Bu bir sohbet motoru demosu/iskeletidir, üretim kalitesinde bir
 *  dil modeli değildir.
 * ══════════════════════════════════════════════════════════════ */

#define CHAT_MAX_PROMPT_TOKENS   256
#define CHAT_MAX_REPLY_TOKENS    40
#define CHAT_EOS_TOKEN_ID        3   /* <eos> */
#define CHAT_UNK_TOKEN_ID        1   /* <unk> */

static void chat_print_banner(const HTLLMModel* model) {
    printf("\n%s%s", CLR_BOLD, CLR_CYAN);
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║   HT-LLM Sohbet Modu (Chat Mode)                             ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║   Yazın ve Enter'a basın. Çıkmak için: /exit veya /quit      ║\n");
    printf("  ║   Bağlamı sıfırlamak için: /reset                            ║\n");
    printf("  ║   Not: kelime haznesi (%4u kelime) dışındaki kelimeler       ║\n", model->tokenizer->vocab_size);
    printf("  ║   <unk> olarak kodlanır. (--resume ile checkpoint'ten         ║\n");
    printf("  ║   öğrenilmiş vocab yüklenmediyse sabit prototip liste kullanılır.) ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", CLR_RESET);
}

static void chat_run(HTLLMModel* model) {
    chat_print_banner(model);

    char line[2048];
    uint32_t token_ids[CHAT_MAX_PROMPT_TOKENS];
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    if (!logits) {
        fprintf(stderr, "%sHATA: Bellek ayrılamadı.%s\n", CLR_RED, CLR_RESET);
        return;
    }

    for (;;) {
        printf("%s%sSen>%s ", CLR_BOLD, CLR_GREEN, CLR_RESET);
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;
        }

        /* Trim trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (len == 0) continue;

        if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) {
            break;
        }
        if (strcmp(line, "/reset") == 0) {
            context_reset(&model->context);
            printf("%s  [Bağlam sıfırlandı]%s\n", CLR_YELLOW, CLR_RESET);
            continue;
        }

        int n_tokens = tokenizer_encode(model->tokenizer, line, token_ids, CHAT_MAX_PROMPT_TOKENS);
        if (n_tokens <= 0) {
            printf("%s  [Boş girdi, tokenize edilemedi]%s\n", CLR_YELLOW, CLR_RESET);
            continue;
        }

        /* Prompt tokenlarını sırayla işleyip bağlamı (EMA context vector)
         * güncelle; son token'dan itibaren otoregresif üretime başla. */
        uint32_t last_token = token_ids[0];
        for (int i = 0; i < n_tokens; i++) {
            htllm_forward(model, token_ids[i], logits);
            last_token = token_ids[i];
        }

        printf("%s%sHT-LLM>%s ", CLR_BOLD, CLR_MAGENTA, CLR_RESET);

        int produced = 0;
        for (int t = 0; t < CHAT_MAX_REPLY_TOKENS; t++) {
            uint32_t next = htllm_generate_next(model, last_token);
            if (next == CHAT_EOS_TOKEN_ID) break;
            const char* word = tokenizer_decode(model->tokenizer, next);
            /* next != CHAT_UNK_TOKEN_ID kapsar id==1 (<unk>); ayrıca
             * "<unused>" (henüz corpus'tan doldurulmamış slotlar) ve
             * tokenizer vocab_size'ın ötesindeki her id de "<unk>"
             * metnine decode olur — hepsini atla. */
            if (next != CHAT_UNK_TOKEN_ID &&
                strcmp(word, "<unk>") != 0 &&
                strcmp(word, "<unused>") != 0) {
                printf("%s ", word);
                produced++;
            }
            last_token = next;
        }
        if (!produced) {
            printf("%s(sessiz kaldı — <unk>/<eos> döngüsü)%s", CLR_DIM, CLR_RESET);
        }
        printf("\n\n");
    }

    free(logits);
    printf("%s  Sohbet oturumu sonlandı.%s\n", CLR_YELLOW, CLR_RESET);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 1: Yönlendirme Yolu Görselleştirmesi
 *  Farklı semantik alanlardaki tokenlar için ağaç traversal'ını gösterir.
 * ══════════════════════════════════════════════════════════════ */

static void demo_routing_visualization(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 1: Yönlendirme Yolu Görselleştirmesi%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    /* Test tokens from different semantic domains */
    const char* test_words[] = {
        "fizik",        /* Science */
        "bilgisayar",   /* Technology */
        "yemek",        /* Daily life */
        "matematik",    /* Math */
    };
    int n_tests = 4;

    for (int t = 0; t < n_tests; t++) {
        printf("\n%s  ── Token: \"%s\" ──%s\n", CLR_YELLOW, test_words[t], CLR_RESET);

        /* Encode token */
        uint32_t token_ids[4];
        int n_tok = tokenizer_encode(model->tokenizer, test_words[t], token_ids, 4);

        if (n_tok >= 2) {  /* At least BOS + token */
            /* Get embedding */
            alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
            embedding_lookup(model->embeddings, token_ids[1], emb);

            /* Update context */
            context_update(&model->context, emb);

            /* Visualize routing */
            tree_print_path(model->root, model->context.vector);

            /* Show active params */
            float logits_buf[VOCAB_SIZE];
            htllm_forward(model, token_ids[1], logits_buf);

            printf("  %sAktif parametre: %lu / %lu (%%%.2f seyreklik)%s\n",
                   CLR_GREEN,
                   (unsigned long)model->active_params_last,
                   (unsigned long)model->total_params,
                   100.0 * (1.0 - (double)model->active_params_last / (double)model->total_params),
                   CLR_RESET);
        }
    }

    /* Reset context for next demo */
    context_reset(&model->context);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 2: Bağlam Takibi ve Topic Drift
 * ══════════════════════════════════════════════════════════════ */

static void demo_context_tracking(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 2: Bağlam Takibi & Topic Drift Tespiti%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    context_reset(&model->context);

    /* Sequence of tokens: Science → Science → DRIFT → Daily life */
    const char* sequence[] = {
        "fizik",      /* Science context */
        "enerji",     /* Still science */
        "atom",       /* Still science */
        "formül",     /* Still science */
        "yemek",      /* TOPIC DRIFT! Daily life */
        "tarif",      /* Daily life */
        "su",         /* Daily life */
    };
    int n_seq = 7;

    printf("\n  Token Dizisi (EMA α=%.2f, drift eşiği=%.2f):\n",
           model->context.alpha, TOPIC_DRIFT_THRESH);
    printf("  %-12s %-12s %-12s %-8s\n",
           "Token", "Benzerlik", "Durum", "Drift?");
    printf("  ────────── ────────── ────────── ──────\n");

    int prev_drift_count = model->context.drift_events;

    for (int i = 0; i < n_seq; i++) {
        uint32_t token_ids[4];
        int n_tok = tokenizer_encode(model->tokenizer, sequence[i], token_ids, 4);
        if (n_tok < 2) continue;

        /* Get embedding */
        alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
        embedding_lookup(model->embeddings, token_ids[1], emb);

        /* Check similarity before update */
        float sim = context_similarity(&model->context, emb);
        int drift_detected = context_detect_topic_drift(&model->context, emb);

        /* Update context */
        context_update(&model->context, emb);

        int new_drift = model->context.drift_events > prev_drift_count;
        prev_drift_count = model->context.drift_events;

        printf("  %-12s %-12.4f %-12s %s%-8s%s\n",
               sequence[i],
               sim,
               model->context.initialized ? "Aktif" : "Başlatılıyor",
               new_drift ? CLR_RED : "",
               drift_detected ? "DRIFT!" : "─",
               new_drift ? CLR_RESET : "");
    }

    printf("\n  %sToplam drift olayı: %d%s\n",
           CLR_YELLOW, model->context.drift_events, CLR_RESET);

    context_reset(&model->context);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 3: Token Üretimi
 * ══════════════════════════════════════════════════════════════ */

static void demo_generation(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 3: Token Üretimi (Eğitimsiz / Rastgele Ağırlıklar)%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    printf("\n  %sNot: Ağırlıklar rastgele başlatıldı — anlamlı metin beklenmemeli.%s\n",
           CLR_DIM, CLR_RESET);
    printf("  %sAmaç: yönlendirme ve latency metriklerini doğrulamak.%s\n\n",
           CLR_DIM, CLR_RESET);

    context_reset(&model->context);
    htllm_generate(model, "fizik enerji atom", 10);

    context_reset(&model->context);
    htllm_generate(model, "bilgisayar algoritma veri", 10);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 4: Latency Benchmark
 * ══════════════════════════════════════════════════════════════ */

static void demo_benchmark(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 4: Latency Benchmark%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    context_reset(&model->context);

    int n_warmup = 50;
    int n_bench  = 500;
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    if (!logits) return;

    /* Warmup */
    printf("\n  Isınma: %d token...\n", n_warmup);
    for (int i = 0; i < n_warmup; i++) {
        uint32_t tok = (uint32_t)(i % 128);
        htllm_forward(model, tok, logits);
    }

    /* Benchmark */
    printf("  Benchmark: %d token...\n", n_bench);

    double min_ms = 1e9, max_ms = 0, total_ms = 0;
    uint64_t total_active = 0;

    for (int i = 0; i < n_bench; i++) {
        uint32_t tok = (uint32_t)(i % 128);

        double t0 = get_time_ms();
        htllm_forward(model, tok, logits);
        double t1 = get_time_ms();

        double elapsed = t1 - t0;
        total_ms += elapsed;
        total_active += model->active_params_last;

        if (elapsed < min_ms) min_ms = elapsed;
        if (elapsed > max_ms) max_ms = elapsed;
    }

    double avg_ms = total_ms / n_bench;
    double tokens_per_sec = 1000.0 / avg_ms;
    double avg_active = (double)total_active / n_bench;
    double avg_sparsity = 100.0 * (1.0 - avg_active / (double)model->total_params);

    printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║               Benchmark Sonuçları                      ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Toplam token       : %d                              ║\n", n_bench);
    printf("  ║  Ortalama latency   : %8.3f ms/token                 ║\n", avg_ms);
    printf("  ║  Min latency        : %8.3f ms                       ║\n", min_ms);
    printf("  ║  Max latency        : %8.3f ms                       ║\n", max_ms);
    printf("  ║  Token/saniye       : %8.1f tok/s                    ║\n", tokens_per_sec);
    printf("  ║  Ort. aktif param   : %'.0f                           ║\n", avg_active);
    printf("  ║  Ort. seyreklik     : %8.2f%%                         ║\n", avg_sparsity);
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║                                                        ║\n");
    printf("  ║  %s── Dense LLM Karşılaştırma (Tahmini) ──%s             ║\n",
           CLR_YELLOW, CLR_RESET);
    printf("  ║                                                        ║\n");
    printf("  ║  Metrik              HT-LLM         Dense 7B (CPU)    ║\n");
    printf("  ║  ──────────────────  ──────────────  ────────────────  ║\n");
    printf("  ║  Latency/token       %6.2f ms        ~2000 ms         ║\n", avg_ms);
    printf("  ║  Token/saniye        %6.1f tok/s     ~0.5 tok/s       ║\n", tokens_per_sec);
    printf("  ║  RAM aktarımı        ~%.0f KB          ~14 GB           ║\n",
           avg_active * sizeof(float) / 1024.0);
    printf("  ║  Aktivasyon          %%%.1f              %%100            ║\n",
           100.0 - avg_sparsity);
    printf("  ║                                                        ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n");

    free(logits);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 5: Çoklu Uzman Aktivasyonu (Polysemy)
 * ══════════════════════════════════════════════════════════════ */

static void demo_multi_expert(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 5: Çoklu Uzman Aktivasyonu (Top-k Routing)%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    context_reset(&model->context);

    printf("\n  Farklı bağlamlarda aynı token için routing davranışı:\n\n");

    /* Context 1: Science → route "enerji" */
    printf("  %s── Bağlam: Bilim ──%s\n", CLR_MAGENTA, CLR_RESET);

    const char* sci_context[] = {"fizik", "atom", "molekül"};
    for (int i = 0; i < 3; i++) {
        uint32_t ids[4];
        int n = tokenizer_encode(model->tokenizer, sci_context[i], ids, 4);
        if (n >= 2) {
            alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
            embedding_lookup(model->embeddings, ids[1], emb);
            context_update(&model->context, emb);
        }
    }

    /* Show routing for "enerji" in science context */
    {
        uint32_t ids[4];
        int n = tokenizer_encode(model->tokenizer, "enerji", ids, 4);
        if (n >= 2) {
            alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
            embedding_lookup(model->embeddings, ids[1], emb);
            context_update(&model->context, emb);

            TreeNode* leaves[TOP_K_MAX * TOP_K_MAX];
            float     weights[TOP_K_MAX * TOP_K_MAX];
            int       n_leaves = 0;
            tree_route_full(model->root, model->context.vector,
                          leaves, weights, &n_leaves);

            printf("  \"enerji\" → %d uzman aktif:", n_leaves);
            for (int i = 0; i < n_leaves; i++) {
                printf(" [Uzman#%u: w=%.3f]", leaves[i]->expert->expert_id, weights[i]);
            }
            printf("\n");
        }
    }

    /* Context 2: Daily life → route "enerji" (different meaning!) */
    context_reset(&model->context);
    printf("\n  %s── Bağlam: Günlük Yaşam ──%s\n", CLR_MAGENTA, CLR_RESET);

    const char* daily_context[] = {"yemek", "su", "kahve"};
    for (int i = 0; i < 3; i++) {
        uint32_t ids[4];
        int n = tokenizer_encode(model->tokenizer, daily_context[i], ids, 4);
        if (n >= 2) {
            alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
            embedding_lookup(model->embeddings, ids[1], emb);
            context_update(&model->context, emb);
        }
    }

    {
        uint32_t ids[4];
        int n = tokenizer_encode(model->tokenizer, "enerji", ids, 4);
        if (n >= 2) {
            alignas(CACHE_LINE_BYTES) float emb[EMBEDDING_DIM];
            embedding_lookup(model->embeddings, ids[1], emb);
            context_update(&model->context, emb);

            TreeNode* leaves[TOP_K_MAX * TOP_K_MAX];
            float     weights[TOP_K_MAX * TOP_K_MAX];
            int       n_leaves = 0;
            tree_route_full(model->root, model->context.vector,
                          leaves, weights, &n_leaves);

            printf("  \"enerji\" → %d uzman aktif:", n_leaves);
            for (int i = 0; i < n_leaves; i++) {
                printf(" [Uzman#%u: w=%.3f]", leaves[i]->expert->expert_id, weights[i]);
            }
            printf("\n");
        }
    }

    printf("\n  %s→ Farklı bağlamlar, aynı kelime için farklı uzmanları aktive eder.%s\n",
           CLR_GREEN, CLR_RESET);

    context_reset(&model->context);
}

/* ══════════════════════════════════════════════════════════════
 *  Demo 6: Eğitim (Training) — Next-Token Prediction
 *  Küçük bir token dizisi üzerinde cross-entropy + SGD ile
 *  loss'un zaman içinde düştüğünü gösterir.
 * ══════════════════════════════════════════════════════════════ */

static void demo_training(HTLLMModel* model) {
    printf("\n%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);
    printf("%s  DEMO 6: Eğitim (Training) — Next-Token Prediction%s\n",
           CLR_BOLD, CLR_RESET);
    printf("%s═══════════════════════════════════════════════════════════%s\n",
           CLR_CYAN, CLR_RESET);

    const char* corpus = "fizik bilgisayar enerji matematik fizik enerji "
                          "bilgisayar matematik fizik bilgisayar enerji matematik";

    uint32_t token_ids[64];
    int n_tokens = tokenizer_encode(model->tokenizer, corpus, token_ids, 64);

    printf("\n  Eğitim korpusu: \"%s\"\n", corpus);
    printf("  Kodlanmış token sayısı: %d\n", n_tokens);

    if (n_tokens < 2) {
        printf("  %sYetersiz token, eğitim atlandı.%s\n", CLR_RED, CLR_RESET);
        return;
    }

    TrainConfig config = {
        .learning_rate = 0.05f,
        .epochs        = 5,
        .verbose       = 0
    };

    htllm_train_sequence(model, token_ids, n_tokens, &config);

    printf("\n  %s→ Loss'un epoch'lar arasında düşmesi, ağırlıkların%s\n", CLR_GREEN, CLR_RESET);
    printf("  %s  next-token prediction görevine uyum sağladığını gösterir.%s\n",
           CLR_GREEN, CLR_RESET);

    context_reset(&model->context);
}

/* ══════════════════════════════════════════════════════════════
 *  Checkpoint Epoch Callback
 *  --checkpoint verildiğinde her --ckpt-every epoch'ta (ve resumable
 *  eğitim fonksiyonlarının son epoch'unda) çağrılır.
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    HTLLMModel* model;
    const char* path;
    uint32_t    seed;
    int         every;
    int         target_epochs;
} CkptCbCtx;

static void htllm_epoch_checkpoint_cb(int epoch, float avg_loss,
                                      uint64_t global_step, void* user_data) {
    CkptCbCtx* ctx = (CkptCbCtx*)user_data;
    if (!ctx || !ctx->path) return;

    int is_last = (epoch >= ctx->target_epochs);
    /* every <= 1 ⇒ her epoch'ta kaydet (varsayılan davranış).
     * every > 1  ⇒ sadece "every"nin katı olan epoch'larda kaydet.
     * Son epoch her zaman kaydedilir (ckpt-every'e bakılmaksızın),
     * böylece eğitim biterken en güncel ağırlıklar diskte olur. */
    if (!is_last && ctx->every > 1 && (epoch % ctx->every != 0)) return;

    TrainState state = {
        .global_step   = global_step,
        .epoch         = (uint32_t)epoch,
        .last_avg_loss = avg_loss
    };
    checkpoint_save(ctx->model, ctx->path, ctx->seed, &state);
}

/* ══════════════════════════════════════════════════════════════
 *  main — Run all demos
 * ══════════════════════════════════════════════════════════════ */

int main(int argc, char* argv[]) {
    uint32_t seed = 42;

    /* ── CLI Argümanları ──
     * --train <dosya>   : belirtilen dataset dosyasıyla eğit ve çık (demo'lar atlanır)
     * --epochs <N>      : epoch sayısı (varsayılan 5)
     * --lr <X>          : öğrenim oranı (varsayılan 0.05)
     * --verbose         : her N satırda ilerleme yazdır
     */
    const char* train_file = NULL;
    int   train_epochs = 5;
    float train_lr = 0.05f;
    int   train_verbose = 0;
    int   train_jsonl = 0;
    const char* train_field = "text";
    const char* ckpt_save_path = NULL;   /* --checkpoint: her ckpt_every epoch'ta buraya kaydet */
    const char* ckpt_resume_path = NULL; /* --resume: bu dosyadan ağırlık+ilerleme durumu yükle */
    int   ckpt_every = 1;                /* --ckpt-every: kaç epoch'ta bir kaydet */
    int   chat_mode = 0;                 /* --chat: interaktif sohbet REPL'i başlat */
    const char* chat_load_path = NULL;   /* --chat ile birlikte --resume ağırlık yükler */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = 1;
        } else if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_file = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            train_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            train_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            train_verbose = 1;
        } else if (strcmp(argv[i], "--jsonl") == 0) {
            train_jsonl = 1;
        } else if (strcmp(argv[i], "--field") == 0 && i + 1 < argc) {
            train_field = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            ckpt_save_path = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            ckpt_resume_path = argv[++i];
        } else if (strcmp(argv[i], "--ckpt-every") == 0 && i + 1 < argc) {
            ckpt_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Kullanım: %s [--train dosya] [--epochs N] [--lr X] [--jsonl] [--field ad] [--verbose]\n", argv[0]);
            printf("               [--checkpoint dosya] [--resume dosya] [--ckpt-every N]\n");
            printf("               [--chat [--resume dosya]]\n");
            printf("  --train <dosya>     : dataset dosyasıyla eğit (belirtilmezse demo modu çalışır)\n");
            printf("  --epochs <N>        : TOPLAM hedef epoch sayısı (varsayılan: 5; --resume ile\n");
            printf("                        checkpoint'teki epoch'tan bu sayıya kadar devam eder)\n");
            printf("  --lr <X>            : öğrenim oranı (varsayılan: 0.05)\n");
            printf("  --jsonl             : dosyayı JSONL olarak oku (her satır bir JSON nesnesi)\n");
            printf("                        .jsonl uzantılı dosyalarda otomatik açılır\n");
            printf("  --field <ad>        : JSONL modunda okunacak alan adı (varsayılan: \"text\")\n");
            printf("  --checkpoint <dosya>: her --ckpt-every epoch'ta ağırlıkları bu dosyaya kaydet\n");
            printf("  --resume <dosya>    : bu checkpoint dosyasından ağırlık+ilerlemeyi yükleyip\n");
            printf("                        kaldığı epoch'tan eğitime devam et (--train ile), ya da\n");
            printf("                        --chat ile birlikte kullanıldığında sohbet öncesi bu\n");
            printf("                        checkpoint'teki eğitilmiş ağırlıkları yükler\n");
            printf("  --ckpt-every <N>    : kaç epoch'ta bir checkpoint kaydedilsin (varsayılan: 1)\n");
            printf("  --verbose           : eğitim sırasında ilerleme yazdır\n");
            printf("  --chat              : demo yerine interaktif sohbet (REPL) modunu başlat\n");
            return 0;
        }
    }

    if (chat_mode) {
        chat_load_path = ckpt_resume_path;
    }

    /* .jsonl uzantısı varsa otomatik JSONL modu */
    if (train_file && !train_jsonl) {
        size_t len = strlen(train_file);
        if (len > 6 && strcmp(train_file + len - 6, ".jsonl") == 0) {
            train_jsonl = 1;
        }
    }

    printf("\n%s%s", CLR_BOLD, CLR_CYAN);
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  ██╗  ██╗████████╗    ██╗     ██╗     ███╗   ███╗          ║\n");
    printf("  ║  ██║  ██║╚══██╔══╝    ██║     ██║     ████╗ ████║          ║\n");
    printf("  ║  ███████║   ██║   ███╗██║     ██║     ██╔████╔██║          ║\n");
    printf("  ║  ██╔══██║   ██║   ╚══╝██║     ██║     ██║╚██╔╝██║          ║\n");
    printf("  ║  ██║  ██║   ██║       ███████╗███████╗██║ ╚═╝ ██║          ║\n");
    printf("  ║  ╚═╝  ╚═╝   ╚═╝       ╚══════╝╚══════╝╚═╝     ╚═╝          ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║  Hierarchical Tree Language Model — Prototype v1.0          ║\n");
    printf("  ║  CPU-Focused Sparse-Activation · O(log N) Routing           ║\n");
    printf("  ║  Designer: Musab Küçük                                      ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n");
    printf("%s\n", CLR_RESET);

    /* ── Create Model ── */
    printf("%s  [1/6] Model oluşturuluyor...%s\n", CLR_YELLOW, CLR_RESET);
    HTLLMModel* model = htllm_create(seed);
    if (!model) {
        fprintf(stderr, "%sHATA: Model oluşturulamadı!%s\n", CLR_RED, CLR_RESET);
        return 1;
    }

    /* ── --chat modu: interaktif REPL, demo/eğitim atlanır ── */
    if (chat_mode) {
        if (chat_load_path) {
            printf("%s  [Chat] Checkpoint yükleniyor: '%s'...%s\n", CLR_YELLOW, chat_load_path, CLR_RESET);
            if (!checkpoint_load(model, chat_load_path, NULL)) {
                fprintf(stderr, "%sHATA: Checkpoint yüklenemedi, rastgele başlatılmış ağırlıklarla devam ediliyor.%s\n",
                        CLR_RED, CLR_RESET);
            } else {
                printf("%s  [Chat] Ağırlıklar yüklendi.%s\n", CLR_GREEN, CLR_RESET);
            }
        } else {
            printf("%s  [Chat] Uyarı: --resume verilmedi, model rastgele başlatılmış ağırlıklarla çalışıyor.%s\n",
                   CLR_YELLOW, CLR_RESET);
        }
        chat_run(model);
        htllm_destroy(model);
        return 0;
    }

    /* ── --train modu: dataset dosyasıyla eğit ve çık ── */
    if (train_file) {
        TrainConfig config = {
            .learning_rate = train_lr,
            .epochs        = train_epochs,
            .verbose       = train_verbose
        };

        int      start_epoch = 0;
        uint64_t start_global_step = 0;

        if (ckpt_resume_path) {
            printf("%s  [Resume] '%s' yükleniyor...%s\n", CLR_YELLOW, ckpt_resume_path, CLR_RESET);
            TrainState resumed = {0};
            if (!checkpoint_load(model, ckpt_resume_path, &resumed)) {
                fprintf(stderr, "%sHATA: Checkpoint yüklenemedi, eğitim durduruldu.%s\n", CLR_RED, CLR_RESET);
                htllm_destroy(model);
                return 1;
            }
            /* Not: checkpoint_load() vocab'ı da yükler (v2+), bu yüzden
             * resume durumunda vocab'ı dataset'ten yeniden inşa ETMİYORUZ
             * — aksi halde kelime↔id eşlemesi checkpoint'teki eğitilmiş
             * embedding/expert ağırlıklarıyla uyuşmaz hale gelir. */
            start_epoch = (int)resumed.epoch;
            start_global_step = resumed.global_step;
            printf("%s  [Resume] Epoch %d, adım %llu itibariyle devam ediliyor (hedef: %d epoch).%s\n",
                   CLR_GREEN, start_epoch, (unsigned long long)start_global_step, train_epochs, CLR_RESET);
        } else {
            /* Sıfırdan eğitim: vocab'ı dataset'ten öğren (sabit prototip
             * listenin üzerine yazılır). Böylece dataset'inizde geçen
             * kelimeler (Türkçe/İngilizce fark etmeksizin) <unk>'a
             * düşmek yerine gerçek token id alır. */
            printf("%s  [Vocab] Kelime haznesi dataset'ten öğreniliyor...%s\n", CLR_YELLOW, CLR_RESET);
            tokenizer_build_from_corpus(model->tokenizer, model->arena,
                                        train_file, train_jsonl, train_field);
        }

        if (ckpt_every < 1) ckpt_every = 1;

        CkptCbCtx ckpt_ctx = { model, ckpt_save_path, seed, ckpt_every, train_epochs };
        HtllmEpochCallback on_epoch_end = ckpt_save_path ? htllm_epoch_checkpoint_cb : NULL;

        int ok;
        if (train_jsonl) {
            ok = htllm_train_from_jsonl_resumable(model, train_file, &config, 64, train_field,
                                                   start_epoch, start_global_step,
                                                   on_epoch_end, &ckpt_ctx);
        } else {
            ok = htllm_train_from_file_resumable(model, train_file, &config, 64,
                                                  start_epoch, start_global_step,
                                                  on_epoch_end, &ckpt_ctx);
        }
        /* Not: htllm_epoch_checkpoint_cb, epoch == target_epochs olduğunda
         * (--ckpt-every'e bakılmaksızın) her zaman kaydeder — böylece
         * eğitim biterken en güncel ağırlıklar mutlaka diskte olur. */

        htllm_destroy(model);
        return ok ? 0 : 1;
    }

    /* ── Arena Stats ── */
    printf("\n%s  [2/6] Bellek istatistikleri%s\n", CLR_YELLOW, CLR_RESET);
    arena_print_stats(model->arena);

    /* ── Model Stats ── */
    printf("\n%s  [3/6] Model istatistikleri%s\n", CLR_YELLOW, CLR_RESET);
    htllm_print_stats(model);

    /* ── Demo 1: Routing ── */
    printf("\n%s  [4/6] Yönlendirme demosu%s\n", CLR_YELLOW, CLR_RESET);
    demo_routing_visualization(model);

    /* ── Demo 2: Context & Drift ── */
    printf("\n%s  [5/6] Bağlam ve drift demosu%s\n", CLR_YELLOW, CLR_RESET);
    demo_context_tracking(model);

    /* ── Demo 3: Generation ── */
    demo_generation(model);

    /* ── Demo 4: Benchmark ── */
    demo_benchmark(model);

    /* ── Demo 5: Multi-Expert ── */
    demo_multi_expert(model);

    /* ── Demo 6: Training ── */
    printf("\n%s  [6/7] Eğitim demosu%s\n", CLR_YELLOW, CLR_RESET);
    demo_training(model);

    /* ── Cleanup ── */
    printf("\n%s  [7/7] Temizlik...%s\n", CLR_YELLOW, CLR_RESET);
    htllm_destroy(model);

    printf("\n%s%s  ✓ HT-LLM Prototip demosu tamamlandı.%s\n\n",
           CLR_BOLD, CLR_GREEN, CLR_RESET);

    return 0;
}
