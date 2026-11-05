/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Hierarchical Tree Language Model
 *  CPU-Focused Sparse-Activation Language Model Prototype
 *
 *  Target Architecture: x86-64 / ARM CPU (SIMD/AVX-256)
 *  Paradigm: O(log N) Dynamic Sparse Routing
 *
 *  Designer: Musab Küçük
 *  Version:  1.0 (Prototype)
 * ══════════════════════════════════════════════════════════════════
 */

#ifndef HT_LLM_H
#define HT_LLM_H

#include <stdint.h>
#include <stddef.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════════════════════════════════════════════
 *  Model Configuration Constants
 * ══════════════════════════════════════════════════════════════ */

#define EMBEDDING_DIM        160     /* Boyut: d = 160 (640 byte FP32)      */
#define MAX_CHILDREN         8       /* Dallanma Faktörü: B = 8              */
#define CACHE_LINE_BYTES     64      /* x86-64 / ARM Cache Line boyutu      */
#define TREE_DEPTH           3       /* Ağaç Derinliği (Root → Leaf)         */
#define VOCAB_SIZE           4096    /* Kelime Hazinesi boyutu               */
#define EXPERT_HIDDEN_DIM    112     /* Mikro-Uzman gizli katman boyutu      */
#define TOP_K_DEFAULT        2       /* Varsayılan Top-k dal sayısı          */
#define TOP_K_MAX            4       /* Maksimum Top-k (yüksek entropi)      */
#define PRUNING_THRESHOLD    0.05f   /* Budama eşiği: P(Dal) < θ            */
#define EMA_ALPHA_DEFAULT    0.85f   /* EMA sönümleme katsayısı α           */
#define TOPIC_DRIFT_THRESH   0.15f   /* Konu kayması tespit eşiği            */
#define SOFTMAX_TEMPERATURE  1.0f    /* Softmax sıcaklık parametresi τ       */
#define ENTROPY_HIGH_THRESH  1.8f    /* Yüksek entropi eşiği (k artırma)     */
#define ENTROPY_LOW_THRESH   0.5f    /* Düşük entropi eşiği (k=1 modu)       */

/* Derived constants */
#define NUM_LEVEL1_NODES     MAX_CHILDREN                           /*   8 */
#define NUM_LEVEL2_NODES     (MAX_CHILDREN * MAX_CHILDREN)          /*  64 */
#define NUM_LEAF_NODES       (MAX_CHILDREN * MAX_CHILDREN * MAX_CHILDREN)  /* 512 */
#define TOTAL_INTERNAL_NODES (1 + NUM_LEVEL1_NODES + NUM_LEVEL2_NODES)    /*  73 */
#define TOTAL_NODES          (TOTAL_INTERNAL_NODES + NUM_LEAF_NODES)      /* 585 */

/* ══════════════════════════════════════════════════════════════
 *  Memory Arena (Contiguous, 64-byte aligned allocator)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    uint8_t* buffer;       /* Contiguous memory block                    */
    size_t   capacity;     /* Total capacity in bytes                    */
    size_t   offset;       /* Current allocation offset                  */
    size_t   peak_offset;  /* High-water mark for stats                  */
    uint32_t alloc_count;  /* Number of allocations made                 */
} MemoryArena;

MemoryArena* arena_create(size_t capacity);
void*        arena_alloc(MemoryArena* arena, size_t size);
void         arena_reset(MemoryArena* arena);
void         arena_destroy(MemoryArena* arena);
void         arena_print_stats(const MemoryArena* arena);

/* ══════════════════════════════════════════════════════════════
 *  Micro-Expert (Leaf FFN)
 *  Architecture: input(128) → W_in → hidden(64) → ReLU → W_out → output(128)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    float*   W_in;         /* [EMBEDDING_DIM × EXPERT_HIDDEN_DIM]       */
    float*   b_in;         /* [EXPERT_HIDDEN_DIM]                        */
    float*   W_out;        /* [EXPERT_HIDDEN_DIM × EMBEDDING_DIM]       */
    float*   b_out;        /* [EMBEDDING_DIM]                            */
    uint32_t expert_id;
    uint32_t hidden_dim;
} MicroExpert;

MicroExpert* expert_create(MemoryArena* arena, uint32_t expert_id);
void         expert_init_random(MicroExpert* expert, uint32_t seed);
void         expert_forward(const MicroExpert* expert, const float* input, float* output);
size_t       expert_param_count(const MicroExpert* expert);

/* ══════════════════════════════════════════════════════════════
 *  Tree Node (64-byte cache-line aligned)
 * ══════════════════════════════════════════════════════════════ */

typedef struct TreeNode {
    alignas(CACHE_LINE_BYTES) float centroid[EMBEDDING_DIM]; /* 512 bytes */
    uint32_t          node_id;
    uint32_t          depth;
    uint32_t          child_count;
    uint32_t          is_leaf;
    struct TreeNode*  children[MAX_CHILDREN];
    MicroExpert*      expert;     /* Non-NULL only for leaf nodes          */
} TreeNode;

/* ══════════════════════════════════════════════════════════════
 *  Routing Result (per routing step)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    int   selected_indices[TOP_K_MAX]; /* Seçilen dal indeksleri            */
    float selected_probs[TOP_K_MAX];   /* Seçilen dalların olasılıkları     */
    int   k;                           /* Aktif dal sayısı                   */
    float entropy;                     /* Softmax dağılımının entropisi      */
    float max_similarity;              /* En yüksek benzerlik skoru          */
} RoutingResult;

TreeNode* tree_create_node(MemoryArena* arena, uint32_t node_id, uint32_t depth);
void      tree_build_random(MemoryArena* arena, TreeNode* root, uint32_t seed);
void      tree_route_step(TreeNode* node, const float* ctx_vector,
                          float temperature, RoutingResult* result);
void      tree_route_full(TreeNode* root, const float* ctx_vector,
                          TreeNode** selected_leaves, float* leaf_weights,
                          int* num_leaves);
void      tree_print_path(TreeNode* root, const float* ctx_vector);
uint32_t  tree_count_nodes(const TreeNode* root);
uint32_t  tree_count_leaves(const TreeNode* root);

/* ══════════════════════════════════════════════════════════════
 *  Context State (EMA-based running context vector)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    alignas(CACHE_LINE_BYTES) float vector[EMBEDDING_DIM];
    float   alpha;          /* EMA decay coefficient                     */
    int     initialized;    /* Whether context has been seeded           */
    int     token_count;    /* Total tokens processed                    */
    int     drift_events;   /* Number of topic drift resets              */
} ContextState;

void  context_init(ContextState* ctx, float alpha);
void  context_update(ContextState* ctx, const float* embedding);
int   context_detect_topic_drift(const ContextState* ctx, const float* new_embedding);
void  context_reset(ContextState* ctx);
float context_similarity(const ContextState* ctx, const float* embedding);

/* ══════════════════════════════════════════════════════════════
 *  Embedding Table
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    float*   embeddings;   /* [vocab_size × dim], cache-aligned          */
    uint32_t vocab_size;
    uint32_t dim;
} EmbeddingTable;

EmbeddingTable* embedding_create(MemoryArena* arena, uint32_t vocab_size, uint32_t dim);
void            embedding_init_random(EmbeddingTable* table, uint32_t seed);
void            embedding_lookup(const EmbeddingTable* table, uint32_t token_id, float* out);

/* ══════════════════════════════════════════════════════════════
 *  Tokenizer (Simple whitespace-based, prototype only)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    char**   vocab;        /* Array of word strings                      */
    uint32_t vocab_size;
} Tokenizer;

Tokenizer*  tokenizer_create(MemoryArena* arena, uint32_t vocab_size);
int         tokenizer_encode(const Tokenizer* tok, const char* text,
                             uint32_t* out_ids, int max_tokens);
const char* tokenizer_decode(const Tokenizer* tok, uint32_t token_id);

/* ── Dinamik vocab (corpus'tan öğrenilen kelime haznesi) ──
 * Sabit PROTO_VOCAB'ın üzerine yazar: dosyayı/dosyaları tarar, kelime
 * frekanslarını sayar, en sık geçen (vocab_size - N_SPECIAL) kelimeyi
 * alıp tok->vocab dizisini bunlarla doldurur. Özel tokenlar (<pad>,
 * <unk>, <bos>, <eos>, ...) her zaman 0..9 indekslerinde sabit kalır.
 * txt modunda satır satır, jsonl modunda field_name alanı okunur
 * (field_name NULL ise "text").
 * Başarılıysa öğrenilen kelime sayısını (özel tokenlar hariç), hata
 * durumunda 0 döner. */
int tokenizer_build_from_corpus(Tokenizer* tok, MemoryArena* arena,
                                const char* filepath, int is_jsonl,
                                const char* field_name);

/* tokenizer_build_from_corpus tarafından üretilen vocab'ı checkpoint
 * dosyasına yazmak/okumak için ham erişim. n_words: özel tokenlar
 * dahil toplam dolu slot sayısı (tok->vocab_size ile aynı olabilir,
 * ama arena'da ayrılan dizi VOCAB_SIZE kapasitelidir). */
uint32_t    tokenizer_word_count(const Tokenizer* tok);
const char* tokenizer_word_at(const Tokenizer* tok, uint32_t index);
int         tokenizer_set_word_at(Tokenizer* tok, MemoryArena* arena,
                                  uint32_t index, const char* word);

/* ══════════════════════════════════════════════════════════════
 *  Output Head (Shared projection: embedding_dim → vocab_size)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    float* W_proj;         /* [EMBEDDING_DIM × VOCAB_SIZE]              */
    float* b_proj;         /* [VOCAB_SIZE]                               */
} OutputHead;

OutputHead* output_head_create(MemoryArena* arena);
void        output_head_init_random(OutputHead* head, uint32_t seed);
void        output_head_forward(const OutputHead* head, const float* expert_out, float* logits);

/* ══════════════════════════════════════════════════════════════
 *  HT-LLM Model (Top-level container)
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    MemoryArena*    arena;
    TreeNode*       root;
    EmbeddingTable* embeddings;
    OutputHead*     output_head;
    ContextState    context;
    Tokenizer*      tokenizer;

    /* Model statistics */
    uint64_t total_params;
    uint64_t active_params_last;  /* Params activated in last forward pass */
    uint32_t total_nodes;
    uint32_t leaf_nodes;
    size_t   model_memory_bytes;
} HTLLMModel;

HTLLMModel* htllm_create(uint32_t seed);
void        htllm_forward(HTLLMModel* model, uint32_t token_id, float* logits);
uint32_t    htllm_generate_next(HTLLMModel* model, uint32_t token_id);
void        htllm_generate(HTLLMModel* model, const char* prompt, int max_tokens);
void        htllm_print_stats(const HTLLMModel* model);
void        htllm_destroy(HTLLMModel* model);

/* ══════════════════════════════════════════════════════════════
 *  Training (Next-token prediction, Cross-Entropy + SGD)
 *
 *  htllm_train_step() eğitimi tek bir (input_token → target_token)
 *  çifti üzerinde çalıştırır:
 *    1. htllm_forward ile logits üretir (ileri yayılım zaten yapılan
 *       route + expert seçimini yeniden kullanır)
 *    2. Softmax + cross-entropy loss hesaplar
 *    3. Output head, seçilen mikro-uzman(lar) ve embedding satırı için
 *       gradyanları geri yayar (backprop)
 *    4. SGD ile ağırlıkları günceller (yalnızca aktif/seçilen parametreler)
 *
 *  Not: Bu, seyrek (sparse) MoE mimarisine uygun "sadece aktive olan
 *  uzmanları güncelle" stratejisidir — routing (centroid) ağırlıkları
 *  bu prototipte sabit tutulur, sadece expert + embedding + output head
 *  eğitilir.
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    float learning_rate;
    int   epochs;
    int   verbose;       /* Her adımda loss yazdır */
} TrainConfig;

/* Tek bir (input, target) örneği üzerinde ileri+geri yayılım + SGD adımı.
 * Döndürülen değer: bu adımın cross-entropy loss'u. */
float htllm_train_step(HTLLMModel* model, uint32_t input_token,
                       uint32_t target_token, float learning_rate);

/* Token dizisi üzerinde otoregresif eğitim: (tok[i] → tok[i+1]) çiftleri.
 * n_epochs kez tüm diziyi dolaşır, ortalama loss'u her epoch sonunda yazdırır. */
void htllm_train_sequence(HTLLMModel* model, const uint32_t* token_ids,
                          int n_tokens, const TrainConfig* config);

/* Metin dosyasından dataset okuyup eğitir.
 * Dosya, satır satır cümle/paragraf içerebilir (UTF-8, whitespace-tokenized).
 * Her satır ayrı bir dizi olarak ele alınır (context her satır başında sıfırlanır).
 * max_tokens_per_line: bir satırdan okunacak maksimum token sayısı.
 * Başarılıysa 1, dosya açılamazsa 0 döner. */
int htllm_train_from_file(HTLLMModel* model, const char* filepath,
                          const TrainConfig* config, int max_tokens_per_line);

/* JSONL dataset dosyasından eğitir. Her satır bağımsız bir JSON nesnesi
 * olmalı, örn: {"text": "fizik bilgisayar enerji"}
 * field_name: hangi JSON alanının okunacağı (NULL ise "text" varsayılan).
 * Sadece basit, kaçışsız (\" içermeyen) düz string değerleri destekler —
 * tam bir JSON parser değildir, prototip amaçlı minimal bir ayrıştırıcıdır.
 * Başarılıysa 1, dosya açılamazsa 0 döner. */
int htllm_train_from_jsonl(HTLLMModel* model, const char* filepath,
                           const TrainConfig* config, int max_tokens_per_line,
                           const char* field_name);

/* Epoch sonu geri çağırma imzası: checkpoint kaydı gibi işler için.
 * epoch: az önce tamamlanan epoch numarası (1-tabanlı)
 * avg_loss: o epoch'un ortalama loss'u
 * global_step: eğitim başından beri atılan toplam SGD adımı
 * user_data: çağıranın verdiği opak işaretçi (ör. model_seed + filepath) */
typedef void (*HtllmEpochCallback)(int epoch, float avg_loss,
                                   uint64_t global_step, void* user_data);

/* htllm_train_from_file ile aynıdır, ek olarak:
 *  - start_epoch: eğitime kaçıncı epoch'tan başlanacağı (resume için;
 *    sıfırdan başlamak için 0 verin). config->epochs, TOPLAM hedef
 *    epoch sayısıdır — döngü start_epoch'tan config->epochs'a kadar sürer.
 *  - start_global_step: resume edilen checkpoint'ten gelen adım sayacı;
 *    callback'e doğru global_step raporlamak için kullanılır.
 *  - on_epoch_end: her epoch tamamlandığında çağrılır (NULL olabilir).
 * Başarılıysa 1, dosya hatasında 0 döner. */
int htllm_train_from_file_resumable(HTLLMModel* model, const char* filepath,
                          const TrainConfig* config, int max_tokens_per_line,
                          int start_epoch, uint64_t start_global_step,
                          HtllmEpochCallback on_epoch_end, void* user_data);

int htllm_train_from_jsonl_resumable(HTLLMModel* model, const char* filepath,
                           const TrainConfig* config, int max_tokens_per_line,
                           const char* field_name,
                           int start_epoch, uint64_t start_global_step,
                           HtllmEpochCallback on_epoch_end, void* user_data);

/* ══════════════════════════════════════════════════════════════
 *  Checkpoint / Resume
 *
 *  Eğitilebilir ağırlıkları (embedding, output head, tüm leaf
 *  expert'ler) + eğitim ilerleme durumunu diske yazar/okur.
 *  Ağaç/centroid'ler kaydedilmez: htllm_create() aynı seed ile
 *  çağrıldığında deterministik olarak yeniden üretilir, bu yüzden
 *  yükleme öncesi model MUTLAKA checkpoint'i üreten seed ile
 *  htllm_create() ile oluşturulmuş olmalıdır.
 * ══════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t global_step;   /* Toplam atılan SGD adımı            */
    uint32_t epoch;         /* Son tamamlanan/devam eden epoch     */
    float    last_avg_loss; /* En son raporlanan ortalama loss     */
} TrainState;

/* Modelin eğitilebilir ağırlıklarını + eğitim durumunu filepath'e yazar.
 * model_seed: model->htllm_create() çağrısında kullanılan seed (resume
 * sırasında doğru ağaç/expert eşleşmesi için dosyaya gömülür).
 * state NULL ise global_step/epoch/last_avg_loss 0 olarak kaydedilir.
 * Başarılıysa 1, hata durumunda 0 döner. */
int checkpoint_save(HTLLMModel* model, const char* filepath,
                    uint32_t model_seed, const TrainState* state);

/* filepath'ten ağırlıkları model'e yükler. model, checkpoint'i üreten
 * seed ile ÖNCEDEN htllm_create() ile oluşturulmuş olmalıdır (aksi
 * halde expert_id eşleşmesi/mimari yanlış olur ve yükleme reddedilir).
 * out_state NULL değilse eğitim ilerleme durumu buraya doldurulur
 * (resume: kaldığı epoch/adımdan devam etmek için).
 * Mimari uyumsuzluğunda veya dosya hatasında 0, başarılıysa 1 döner. */
int checkpoint_load(HTLLMModel* model, const char* filepath, TrainState* out_state);

/* ══════════════════════════════════════════════════════════════
 *  Utility / Debug
 * ══════════════════════════════════════════════════════════════ */

/* Portable high-resolution timer */
double get_time_ms(void);

/* Simple xoshiro128+ PRNG (fast, deterministic) */
uint32_t xoshiro128p_next(uint32_t state[4]);
void     xoshiro128p_seed(uint32_t state[4], uint32_t seed);
float    random_uniform(uint32_t state[4]);               /* [0, 1)   */
float    random_normal(uint32_t state[4]);                /* N(0, 1)  */

#ifdef __cplusplus
}
#endif

#endif /* HT_LLM_H */
