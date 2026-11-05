/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Checkpoint Save / Resume Module
 *
 *  Neyi kaydeder:
 *   - Embedding tablosu           [VOCAB_SIZE x EMBEDDING_DIM]
 *   - Output head (W_proj,b_proj) [EMBEDDING_DIM x VOCAB_SIZE] + [VOCAB_SIZE]
 *   - Her leaf micro-expert       W_in,b_in,W_out,b_out (leaf sırasına göre,
 *     ağaç inşa edilirken atanan expert_id 0..NUM_LEAF_NODES-1 sırasıyla)
 *   - Tokenizer vocab'ı (v2+)     [VOCAB_SIZE adet değişken-uzunluklu string]
 *   - Eğitim durumu: global_step, epoch, learning_rate, dosya adı vb.
 *
 *  Neyi kaydetmez (kasıtlı):
 *   - Ağaç yapısı / centroid'ler: tree_build_random() deterministiktir,
 *     aynı seed ile htllm_create() çağrıldığında birebir aynı ağaç ve
 *     aynı expert_id ataması yeniden üretilir. Routing zaten eğitim
 *     dışı (sabit) olduğundan bunları diskte tutmaya gerek yok.
 *
 *  Dosya formatı (little-endian, ham binary):
 *   [MAGIC: "HTLC"][uint32 format_version]
 *   [uint32 embedding_dim][uint32 vocab_size][uint32 expert_hidden_dim]
 *   [uint32 num_leaf_nodes][uint32 model_seed]
 *   [uint64 global_step][uint32 epoch][float last_avg_loss]
 *   -- v2: vocab --
 *   [uint32 n_words] repeated n_words times: [uint32 word_len][word_len byte]
 *   -- payload --
 *   [embeddings: vocab_size * embedding_dim floats]
 *   [W_proj: embedding_dim * vocab_size floats][b_proj: vocab_size floats]
 *   for each leaf (expert_id 0..num_leaf_nodes-1):
 *     [W_in: embedding_dim*hidden][b_in: hidden]
 *     [W_out: hidden*embedding_dim][b_out: embedding_dim]
 *   [uint32 CHECK_MAGIC: "HTLE"]  (dosyanın eksiksiz yazıldığını doğrulamak için)
 *
 *  Mimari uyumsuzluğu (farklı EMBEDDING_DIM/VOCAB_SIZE/hidden/leaf sayısı
 *  ile derlenmiş bir binary) tespit edilirse yükleme reddedilir.
 *  v1 formatındaki eski checkpoint'ler (dinamik vocab öncesi) bu build
 *  ile OKUNAMAZ — format_version uyuşmazlığından reddedilir.
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HTLLM_CKPT_MAGIC   0x434C5448u  /* "HTLC" (little-endian byte order olarak yazılır) */
#define HTLLM_CKPT_ENDMAGIC 0x454C5448u /* "HTLE" */
#define HTLLM_CKPT_VERSION  2u          /* v2: dinamik (corpus'tan öğrenilen) vocab da kaydedilir */

/* TrainState struct tanımı ht_llm.h içinde. */

/* ── Yardımcı: ağaçtaki leaf'leri, tree_build_random() sırasında atanan
 * expert_id (0..NUM_LEAF_NODES-1) sırasına göre bir diziye toplar.
 * Bu sıralama, aynı seed ile yeniden oluşturulan bir modelde de birebir
 * aynıdır (deterministik DFS inşa sırası). ── */
static void collect_leaves_by_expert_id(TreeNode* node, TreeNode** out, uint32_t out_capacity) {
    if (!node) return;
    if (node->is_leaf) {
        if (node->expert && node->expert->expert_id < out_capacity) {
            out[node->expert->expert_id] = node;
        }
        return;
    }
    for (uint32_t i = 0; i < node->child_count; i++) {
        collect_leaves_by_expert_id(node->children[i], out, out_capacity);
    }
}

static int write_u32(FILE* f, uint32_t v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static int write_u64(FILE* f, uint64_t v) { return fwrite(&v, sizeof(v), 1, f) == 1; }
static int write_f32(FILE* f, float v)    { return fwrite(&v, sizeof(v), 1, f) == 1; }
static int write_arr(FILE* f, const float* a, size_t n) {
    return n == 0 || fwrite(a, sizeof(float), n, f) == n;
}

static int read_u32(FILE* f, uint32_t* v) { return fread(v, sizeof(*v), 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, sizeof(*v), 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, sizeof(*v), 1, f) == 1; }
static int read_arr(FILE* f, float* a, size_t n) {
    return n == 0 || fread(a, sizeof(float), n, f) == n;
}

/* ══════════════════════════════════════════════════════════════
 *  checkpoint_save
 *  Başarılıysa 1, hata durumunda 0 döner.
 * ══════════════════════════════════════════════════════════════ */
int checkpoint_save(HTLLMModel* model, const char* filepath,
                     uint32_t model_seed, const TrainState* state) {
    if (!model || !filepath) return 0;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        fprintf(stderr, "  [Checkpoint] HATA: '%s' yazmak için açılamadı.\n", filepath);
        return 0;
    }

    int ok = 1;
    ok &= write_u32(f, HTLLM_CKPT_MAGIC);
    ok &= write_u32(f, HTLLM_CKPT_VERSION);
    ok &= write_u32(f, (uint32_t)EMBEDDING_DIM);
    ok &= write_u32(f, (uint32_t)VOCAB_SIZE);
    ok &= write_u32(f, (uint32_t)EXPERT_HIDDEN_DIM);
    ok &= write_u32(f, (uint32_t)NUM_LEAF_NODES);
    ok &= write_u32(f, model_seed);

    uint64_t global_step = state ? state->global_step : 0;
    uint32_t epoch        = state ? state->epoch : 0;
    float    last_loss    = state ? state->last_avg_loss : 0.0f;
    ok &= write_u64(f, global_step);
    ok &= write_u32(f, epoch);
    ok &= write_f32(f, last_loss);

    /* ── v2: Vocab (tokenizer kelime listesi) ──
     * Her slot: [uint32 word_len][word_len byte, sonlandırıcı olmadan] */
    {
        Tokenizer* tok = model->tokenizer;
        uint32_t n_words = tok ? tok->vocab_size : 0;
        ok &= write_u32(f, n_words);
        for (uint32_t i = 0; i < n_words && ok; i++) {
            const char* w = tok->vocab[i];
            uint32_t wlen = (uint32_t)strlen(w);
            ok &= write_u32(f, wlen);
            if (wlen > 0) {
                ok &= (fwrite(w, 1, wlen, f) == wlen);
            }
        }
    }

    /* Embeddings */
    ok &= write_arr(f, model->embeddings->embeddings,
                     (size_t)VOCAB_SIZE * EMBEDDING_DIM);

    /* Output head */
    OutputHead* head = model->output_head;
    ok &= write_arr(f, head->W_proj, (size_t)EMBEDDING_DIM * VOCAB_SIZE);
    ok &= write_arr(f, head->b_proj, (size_t)VOCAB_SIZE);

    /* Leaf experts, expert_id sırasına göre */
    TreeNode** leaves = (TreeNode**)calloc(NUM_LEAF_NODES, sizeof(TreeNode*));
    if (!leaves) {
        fprintf(stderr, "  [Checkpoint] HATA: bellek ayrılamadı.\n");
        fclose(f);
        return 0;
    }
    collect_leaves_by_expert_id(model->root, leaves, NUM_LEAF_NODES);

    for (uint32_t i = 0; i < NUM_LEAF_NODES && ok; i++) {
        if (!leaves[i] || !leaves[i]->expert) {
            fprintf(stderr, "  [Checkpoint] HATA: expert_id=%u için leaf bulunamadı.\n", i);
            ok = 0;
            break;
        }
        MicroExpert* ex = leaves[i]->expert;
        ok &= write_arr(f, ex->W_in,  (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM);
        ok &= write_arr(f, ex->b_in,  (size_t)EXPERT_HIDDEN_DIM);
        ok &= write_arr(f, ex->W_out, (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM);
        ok &= write_arr(f, ex->b_out, (size_t)EMBEDDING_DIM);
    }
    free(leaves);

    ok &= write_u32(f, HTLLM_CKPT_ENDMAGIC);

    fclose(f);

    if (!ok) {
        fprintf(stderr, "  [Checkpoint] HATA: '%s' yazılırken hata oluştu.\n", filepath);
        remove(filepath);
        return 0;
    }

    printf("  [Checkpoint] ✓ Kaydedildi: %s  (adım=%llu, epoch=%u, loss=%.4f)\n",
           filepath, (unsigned long long)global_step, epoch, last_loss);
    return 1;
}

/* ══════════════════════════════════════════════════════════════
 *  checkpoint_load
 *  model, checkpoint'in kaydedildiği seed ile htllm_create() ile
 *  ÖNCEDEN oluşturulmuş olmalı (ağaç/centroid'ler ve expert_id
 *  ataması bu şekilde checkpoint ile birebir eşleşir).
 *  out_state doluysa (NULL değilse) eğitim ilerleme durumu buraya
 *  yazılır (resume için).
 *  Başarılıysa 1, mimari uyumsuzluğu veya dosya hatasında 0 döner. */
int checkpoint_load(HTLLMModel* model, const char* filepath, TrainState* out_state) {
    if (!model || !filepath) return 0;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "  [Checkpoint] HATA: '%s' okumak için açılamadı.\n", filepath);
        return 0;
    }

    uint32_t magic = 0, version = 0;
    uint32_t emb_dim = 0, vocab = 0, hidden = 0, n_leaf = 0, seed = 0;
    uint64_t global_step = 0;
    uint32_t epoch = 0;
    float    last_loss = 0.0f;

    int ok = 1;
    ok &= read_u32(f, &magic);
    if (!ok || magic != HTLLM_CKPT_MAGIC) {
        fprintf(stderr, "  [Checkpoint] HATA: '%s' geçerli bir HT-LLM checkpoint dosyası değil.\n", filepath);
        fclose(f);
        return 0;
    }
    ok &= read_u32(f, &version);
    ok &= read_u32(f, &emb_dim);
    ok &= read_u32(f, &vocab);
    ok &= read_u32(f, &hidden);
    ok &= read_u32(f, &n_leaf);
    ok &= read_u32(f, &seed);
    ok &= read_u64(f, &global_step);
    ok &= read_u32(f, &epoch);
    ok &= read_f32(f, &last_loss);

    if (!ok) {
        fprintf(stderr, "  [Checkpoint] HATA: dosya başlığı okunamadı (bozuk dosya).\n");
        fclose(f);
        return 0;
    }

    if (version != HTLLM_CKPT_VERSION) {
        fprintf(stderr,
            "  [Checkpoint] HATA: format sürümü uyumsuz (dosya=%u, beklenen=%u).\n"
            "    v1 checkpoint'ler (dinamik vocab öncesi) bu build ile uyumsuzdur;\n"
            "    modeli bu checkpoint'ten devam ettirmek için sıfırdan eğitmeniz gerekir.\n",
            version, HTLLM_CKPT_VERSION);
        fclose(f);
        return 0;
    }

    if (emb_dim != EMBEDDING_DIM || vocab != VOCAB_SIZE ||
        hidden != EXPERT_HIDDEN_DIM || n_leaf != NUM_LEAF_NODES) {
        fprintf(stderr,
            "  [Checkpoint] HATA: mimari uyumsuz.\n"
            "    dosya:    dim=%u vocab=%u hidden=%u leaf=%u\n"
            "    beklenen: dim=%d vocab=%d hidden=%d leaf=%d\n",
            emb_dim, vocab, hidden, n_leaf,
            EMBEDDING_DIM, VOCAB_SIZE, EXPERT_HIDDEN_DIM, NUM_LEAF_NODES);
        fclose(f);
        return 0;
    }

    /* ── v2: Vocab yükle ── */
    {
        uint32_t n_words = 0;
        ok &= read_u32(f, &n_words);
        if (!ok) {
            fprintf(stderr, "  [Checkpoint] HATA: vocab başlığı okunamadı.\n");
            fclose(f);
            return 0;
        }
        if (n_words != model->tokenizer->vocab_size) {
            fprintf(stderr,
                "  [Checkpoint] HATA: vocab boyutu uyumsuz (dosya=%u, beklenen=%u).\n",
                n_words, model->tokenizer->vocab_size);
            fclose(f);
            return 0;
        }
        char wbuf[256];
        for (uint32_t i = 0; i < n_words && ok; i++) {
            uint32_t wlen = 0;
            ok &= read_u32(f, &wlen);
            if (!ok || wlen >= sizeof(wbuf)) { ok = 0; break; }
            if (wlen > 0) {
                ok &= (fread(wbuf, 1, wlen, f) == wlen);
            }
            wbuf[wlen] = '\0';
            if (ok) {
                tokenizer_set_word_at(model->tokenizer, model->arena, i, wbuf);
            }
        }
        if (!ok) {
            fprintf(stderr, "  [Checkpoint] HATA: vocab okunurken hata oluştu (bozuk dosya).\n");
            fclose(f);
            return 0;
        }
    }

    /* Embeddings */
    ok &= read_arr(f, model->embeddings->embeddings, (size_t)VOCAB_SIZE * EMBEDDING_DIM);

    /* Output head */
    OutputHead* head = model->output_head;
    ok &= read_arr(f, head->W_proj, (size_t)EMBEDDING_DIM * VOCAB_SIZE);
    ok &= read_arr(f, head->b_proj, (size_t)VOCAB_SIZE);

    TreeNode** leaves = (TreeNode**)calloc(NUM_LEAF_NODES, sizeof(TreeNode*));
    if (!leaves) {
        fprintf(stderr, "  [Checkpoint] HATA: bellek ayrılamadı.\n");
        fclose(f);
        return 0;
    }
    collect_leaves_by_expert_id(model->root, leaves, NUM_LEAF_NODES);

    for (uint32_t i = 0; i < NUM_LEAF_NODES && ok; i++) {
        if (!leaves[i] || !leaves[i]->expert) {
            fprintf(stderr, "  [Checkpoint] HATA: expert_id=%u için leaf bulunamadı (model/checkpoint uyumsuz).\n", i);
            ok = 0;
            break;
        }
        MicroExpert* ex = leaves[i]->expert;
        ok &= read_arr(f, ex->W_in,  (size_t)EMBEDDING_DIM * EXPERT_HIDDEN_DIM);
        ok &= read_arr(f, ex->b_in,  (size_t)EXPERT_HIDDEN_DIM);
        ok &= read_arr(f, ex->W_out, (size_t)EXPERT_HIDDEN_DIM * EMBEDDING_DIM);
        ok &= read_arr(f, ex->b_out, (size_t)EMBEDDING_DIM);
    }
    free(leaves);

    uint32_t end_magic = 0;
    ok &= read_u32(f, &end_magic);
    if (ok && end_magic != HTLLM_CKPT_ENDMAGIC) {
        fprintf(stderr, "  [Checkpoint] UYARI: dosya sonu imzası eşleşmiyor, dosya eksik/bozuk olabilir.\n");
        ok = 0;
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "  [Checkpoint] HATA: '%s' okunurken hata oluştu (ağırlıklar kısmen yüklenmiş olabilir).\n",
                filepath);
        return 0;
    }

    if (seed != 0 && out_state) {
        /* seed bilgisini state ile birlikte kullanan çağıran isterse
         * loglayabilir; burada sadece uyarı amaçlı saklanmıyor. */
    }

    if (out_state) {
        out_state->global_step   = global_step;
        out_state->epoch         = epoch;
        out_state->last_avg_loss = last_loss;
    }

    printf("  [Checkpoint] ✓ Yüklendi: %s  (adım=%llu, epoch=%u, son loss=%.4f, model_seed=%u)\n",
           filepath, (unsigned long long)global_step, epoch, last_loss, seed);
    return 1;
}
