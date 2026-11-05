/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Simple Tokenizer (Prototype)
 *  Whitespace-based word tokenizer with fixed vocabulary.
 *  Production: BPE/SentencePiece ile değiştirilecek.
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include "simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ══════════════════════════════════════════════════════════════
 *  Prototype vocabulary — semantic category tokens
 *  Gerçek üretimde BPE tokenizer kullanılacak.
 * ══════════════════════════════════════════════════════════════ */

static const char* PROTO_VOCAB[] = {
    /* 0-9: Special tokens */
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>",
    "<mask>", "<cls>", "<usr>", "<sys>", "<nl>",

    /* 10-29: Common Turkish words */
    "bir", "ve", "bu", "için", "ile",
    "de", "da", "ne", "çok", "var",
    "ben", "sen", "o", "biz", "siz",
    "onlar", "gibi", "ama", "veya", "hem",

    /* 30-49: Science & Math */
    "fizik", "matematik", "biyoloji", "kimya", "enerji",
    "atom", "hücre", "gen", "protein", "molekül",
    "formül", "denklem", "integral", "türev", "matris",
    "vektör", "tensör", "fonksiyon", "olasılık", "istatistik",

    /* 50-69: Technology */
    "bilgisayar", "yazılım", "donanım", "ağ", "veri",
    "algoritma", "program", "kod", "sistem", "işlemci",
    "bellek", "önbellek", "cpu", "gpu", "ram",
    "sunucu", "veritabanı", "bulut", "yapay", "zeka",

    /* 70-89: Daily life */
    "yemek", "tarif", "su", "ekmek", "peynir",
    "çay", "kahve", "ev", "oda", "bahçe",
    "yol", "araba", "uçak", "tren", "okul",
    "kitap", "kalem", "masa", "sandalye", "pencere",

    /* 90-109: Verbs / Actions */
    "yapmak", "gitmek", "gelmek", "almak", "vermek",
    "bilmek", "görmek", "söylemek", "düşünmek", "anlamak",
    "hesaplamak", "çözmek", "öğrenmek", "öğretmek", "araştırmak",
    "analiz", "sentez", "optimize", "derlemek", "çalıştırmak",

    /* 110-127: Numbers and misc */
    "sıfır", "bir_s", "iki", "üç", "dört",
    "beş", "altı", "yedi", "sekiz", "dokuz",
    "yüz", "bin", "milyon", "milyar", "artı",
    "eksi", "çarpı", "bölü"
};

#define PROTO_VOCAB_SIZE  128

/* ══════════════════════════════════════════════════════════════
 *  Tokenizer Create
 * ══════════════════════════════════════════════════════════════ */

Tokenizer* tokenizer_create(MemoryArena* arena, uint32_t vocab_size) {
    Tokenizer* tok = (Tokenizer*)arena_alloc(arena, sizeof(Tokenizer));
    if (!tok) return NULL;

    /* vocab_size çağıran tarafından VOCAB_SIZE (ör. 4096) olarak verilir.
     * Başlangıçta: 0-9 özel tokenlar + PROTO_VOCAB'daki 118 sabit kelime
     * ile doldurulur, geri kalan slotlar "<unused>" placeholder alır.
     * tokenizer_build_from_corpus() çağrılırsa 10..vocab_size-1 aralığı
     * dataset'ten öğrenilen kelimelerle üzerine yazılır. */
    if (vocab_size < PROTO_VOCAB_SIZE) vocab_size = PROTO_VOCAB_SIZE;
    tok->vocab_size = vocab_size;

    /* Allocate pointer array */
    tok->vocab = (char**)arena_alloc(arena, (size_t)vocab_size * sizeof(char*));
    if (!tok->vocab) return NULL;

    /* Copy static vocabulary strings into arena (0..PROTO_VOCAB_SIZE-1) */
    for (uint32_t i = 0; i < PROTO_VOCAB_SIZE; i++) {
        size_t len = strlen(PROTO_VOCAB[i]) + 1;
        char* word = (char*)arena_alloc(arena, len);
        if (!word) return NULL;
        memcpy(word, PROTO_VOCAB[i], len);
        tok->vocab[i] = word;
    }

    /* Kalan slotları placeholder ile doldur */
    for (uint32_t i = PROTO_VOCAB_SIZE; i < vocab_size; i++) {
        const char* ph = "<unused>";
        size_t len = strlen(ph) + 1;
        char* word = (char*)arena_alloc(arena, len);
        if (!word) return NULL;
        memcpy(word, ph, len);
        tok->vocab[i] = word;
    }

    return tok;
}

/* ══════════════════════════════════════════════════════════════
 *  Tokenizer Encode — text → token IDs
 *  Simple whitespace tokenizer with vocabulary lookup.
 * ══════════════════════════════════════════════════════════════ */

int tokenizer_encode(const Tokenizer* tok, const char* text,
                     uint32_t* out_ids, int max_tokens) {
    if (!tok || !text || !out_ids || max_tokens <= 0) return 0;

    int token_count = 0;
    const char* p = text;

    /* Add BOS token */
    if (token_count < max_tokens) {
        out_ids[token_count++] = 2;  /* <bos> */
    }

    while (*p && token_count < max_tokens) {
        /* Skip whitespace */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Extract word */
        const char* word_start = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        size_t word_len = (size_t)(p - word_start);

        /* Lookup in vocabulary */
        uint32_t token_id = 1;  /* Default: <unk> */
        for (uint32_t i = 0; i < tok->vocab_size; i++) {
            if (strlen(tok->vocab[i]) == word_len &&
                strncmp(tok->vocab[i], word_start, word_len) == 0) {
                token_id = i;
                break;
            }
        }

        out_ids[token_count++] = token_id;
    }

    /* Add EOS token */
    if (token_count < max_tokens) {
        out_ids[token_count++] = 3;  /* <eos> */
    }

    return token_count;
}

/* ══════════════════════════════════════════════════════════════
 *  Tokenizer Decode — token ID → word string
 * ══════════════════════════════════════════════════════════════ */

const char* tokenizer_decode(const Tokenizer* tok, uint32_t token_id) {
    if (!tok || token_id >= tok->vocab_size) {
        return "<unk>";
    }
    return tok->vocab[token_id];
}

/* ══════════════════════════════════════════════════════════════
 *  Dinamik Vocab — Corpus'tan Kelime Haznesi Öğrenme
 *
 *  Yöntem: dosyayı tarar, her kelimeyi (whitespace-split) bir hash
 *  tabloda sayar, sonra frekansa göre azalan sırayla en sık N
 *  kelimeyi seçip tok->vocab[10..] aralığına yazar (0-9 özel
 *  tokenlar için ayrılmıştır ve dokunulmaz).
 * ══════════════════════════════════════════════════════════════ */

#define VOCAB_BUILD_MAX_UNIQUE   65536
#define VOCAB_BUILD_MAX_WORDLEN  63
#define VOCAB_BUILD_N_SPECIAL    10   /* <pad>..<nl>, PROTO_VOCAB[0..9] ile aynı */

typedef struct {
    char     word[VOCAB_BUILD_MAX_WORDLEN + 1];
    uint64_t count;
} WordFreq;

/* Basit lineer-prob hash tablosu (prototip amaçlı, performans kritik değil) */
typedef struct {
    WordFreq* slots;
    uint32_t  capacity;
    uint32_t  n_unique;
} FreqTable;

static uint32_t fnv1a_hash(const char* s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

static void freq_table_add(FreqTable* ft, const char* word, size_t len) {
    if (len == 0 || len > VOCAB_BUILD_MAX_WORDLEN) return;

    uint32_t h = fnv1a_hash(word, len) % ft->capacity;
    for (uint32_t probe = 0; probe < ft->capacity; probe++) {
        uint32_t idx = (h + probe) % ft->capacity;
        WordFreq* slot = &ft->slots[idx];

        if (slot->count == 0) {
            /* Boş slot: yeni kelime */
            memcpy(slot->word, word, len);
            slot->word[len] = '\0';
            slot->count = 1;
            ft->n_unique++;
            return;
        }
        if (strlen(slot->word) == len && strncmp(slot->word, word, len) == 0) {
            slot->count++;
            return;
        }
        /* Çakışma: bir sonraki slot'a bak (linear probing) */
    }
    /* Tablo dolu — kelime atlanır (prototip için kabul edilebilir sınır) */
}

/* Küçük harfe çevirme (ASCII + Türkçe İ/ı/Ş/Ğ/Ü/Ö/Ç basit eşleme).
 * Tam Unicode-doğru değildir (UTF-8 byte düzeyinde çalışır), ama
 * whitespace-tokenized prototip için yeterlidir. */
static void lowercase_ascii_inplace(char* s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') s[i] = (char)(c - 'A' + 'a');
    }
}

/* Bir metin satırından kelimeleri çıkarıp frekans tablosuna ekler. */
static void scan_line_into_freq_table(FreqTable* ft, const char* line) {
    const char* p = line;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        const char* word_start = p;
        while (*p && !isspace((unsigned char)*p)) p++;
        size_t word_len = (size_t)(p - word_start);
        if (word_len == 0 || word_len > VOCAB_BUILD_MAX_WORDLEN) continue;

        char buf[VOCAB_BUILD_MAX_WORDLEN + 1];
        memcpy(buf, word_start, word_len);
        buf[word_len] = '\0';
        lowercase_ascii_inplace(buf, word_len);

        freq_table_add(ft, buf, word_len);
    }
}

/* JSONL satırından basit "field_name": "..." değerini çıkarır (kaçışsız,
 * train.c'deki minimal JSONL parser ile aynı felsefede). out_buf'a
 * yazar, bulunamazsa 0 döner. */
static int extract_jsonl_field(const char* line, const char* field_name,
                                char* out_buf, size_t out_cap) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", field_name);

    const char* key_pos = strstr(line, pattern);
    if (!key_pos) return 0;

    const char* p = key_pos + strlen(pattern);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return 0;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '"') return 0;
    p++;

    size_t n = 0;
    while (*p && *p != '"' && n + 1 < out_cap) {
        out_buf[n++] = *p++;
    }
    out_buf[n] = '\0';
    return n > 0;
}

/* qsort karşılaştırıcı: frekansa göre azalan sıralama */
static int wordfreq_cmp_desc(const void* a, const void* b) {
    const WordFreq* wa = (const WordFreq*)a;
    const WordFreq* wb = (const WordFreq*)b;
    if (wa->count > wb->count) return -1;
    if (wa->count < wb->count) return 1;
    /* Eşitlikte alfabetik sırayla deterministik sonuç */
    return strcmp(wa->word, wb->word);
}

uint32_t tokenizer_word_count(const Tokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

const char* tokenizer_word_at(const Tokenizer* tok, uint32_t index) {
    if (!tok || index >= tok->vocab_size) return NULL;
    return tok->vocab[index];
}

int tokenizer_set_word_at(Tokenizer* tok, MemoryArena* arena,
                          uint32_t index, const char* word) {
    if (!tok || !arena || !word || index >= tok->vocab_size) return 0;
    size_t len = strlen(word) + 1;
    char* copy = (char*)arena_alloc(arena, len);
    if (!copy) return 0;
    memcpy(copy, word, len);
    tok->vocab[index] = copy;
    return 1;
}

int tokenizer_build_from_corpus(Tokenizer* tok, MemoryArena* arena,
                                const char* filepath, int is_jsonl,
                                const char* field_name) {
    if (!tok || !arena || !filepath) return 0;

    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "  [Vocab] HATA: '%s' okumak için açılamadı.\n", filepath);
        return 0;
    }

    FreqTable ft;
    ft.capacity = VOCAB_BUILD_MAX_UNIQUE;
    ft.n_unique = 0;
    ft.slots = (WordFreq*)calloc(ft.capacity, sizeof(WordFreq));
    if (!ft.slots) {
        fprintf(stderr, "  [Vocab] HATA: bellek ayrılamadı.\n");
        fclose(f);
        return 0;
    }

    char line_buf[4096];
    const char* field = field_name ? field_name : "text";
    uint64_t n_lines = 0;

    while (fgets(line_buf, sizeof(line_buf), f)) {
        n_lines++;
        if (is_jsonl) {
            char text_buf[4096];
            if (extract_jsonl_field(line_buf, field, text_buf, sizeof(text_buf))) {
                scan_line_into_freq_table(&ft, text_buf);
            }
        } else {
            scan_line_into_freq_table(&ft, line_buf);
        }
    }
    fclose(f);

    if (ft.n_unique == 0) {
        fprintf(stderr, "  [Vocab] UYARI: '%s' içinde kelime bulunamadı, sabit vocab korunuyor.\n", filepath);
        free(ft.slots);
        return 0;
    }

    /* Sıkıştır: sadece dolu slot'ları ayrı bir diziye topla, sonra sırala */
    WordFreq* dense = (WordFreq*)malloc(ft.n_unique * sizeof(WordFreq));
    if (!dense) {
        fprintf(stderr, "  [Vocab] HATA: bellek ayrılamadı.\n");
        free(ft.slots);
        return 0;
    }
    uint32_t d = 0;
    for (uint32_t i = 0; i < ft.capacity; i++) {
        if (ft.slots[i].count > 0) dense[d++] = ft.slots[i];
    }
    free(ft.slots);

    qsort(dense, d, sizeof(WordFreq), wordfreq_cmp_desc);

    uint32_t max_learnable = tok->vocab_size - VOCAB_BUILD_N_SPECIAL;
    uint32_t n_take = (d < max_learnable) ? d : max_learnable;

    for (uint32_t i = 0; i < n_take; i++) {
        uint32_t slot_idx = VOCAB_BUILD_N_SPECIAL + i;
        size_t len = strlen(dense[i].word) + 1;
        char* copy = (char*)arena_alloc(arena, len);
        if (!copy) break;
        memcpy(copy, dense[i].word, len);
        tok->vocab[slot_idx] = copy;
    }

    /* Kalan slotları (öğrenilenden az ise) placeholder ile doldur, boş
     * pointer bırakmamak için — bu slotlar hiçbir zaman gerçek metinle
     * eşleşmeyecek ama decode/encode güvenli kalır. */
    for (uint32_t i = VOCAB_BUILD_N_SPECIAL + n_take; i < tok->vocab_size; i++) {
        const char* ph = "<unused>";
        size_t len = strlen(ph) + 1;
        char* copy = (char*)arena_alloc(arena, len);
        if (!copy) break;
        memcpy(copy, ph, len);
        tok->vocab[i] = copy;
    }

    printf("  [Vocab] '%s' tarandı: %llu satır, %u benzersiz kelime, %u kelime vocab'a alındı (kapasite: %u).\n",
           filepath, (unsigned long long)n_lines, d, n_take, max_learnable);

    free(dense);
    return (int)n_take;
}
