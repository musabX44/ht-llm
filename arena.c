/*
 * ══════════════════════════════════════════════════════════════════
 *  HT-LLM: Memory Arena Allocator
 *  Contiguous, 64-byte cache-line aligned memory pool.
 *  Zero fragmentation — single free() releases the entire model.
 * ══════════════════════════════════════════════════════════════════
 */

#include "ht_llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════
 *  Arena Create — allocate a single contiguous aligned block
 * ══════════════════════════════════════════════════════════════ */

MemoryArena* arena_create(size_t capacity) {
    MemoryArena* arena = (MemoryArena*)malloc(sizeof(MemoryArena));
    if (!arena) {
        fprintf(stderr, "[Arena] HATA: MemoryArena yapısı tahsis edilemedi.\n");
        return NULL;
    }

    arena->capacity    = capacity;
    arena->offset      = 0;
    arena->peak_offset = 0;
    arena->alloc_count = 0;

    /* posix_memalign: 64-byte aligned contiguous allocation */
    int ret = posix_memalign((void**)&arena->buffer, CACHE_LINE_BYTES, capacity);
    if (ret != 0 || !arena->buffer) {
        fprintf(stderr, "[Arena] HATA: %zu byte bellek tahsis edilemedi (hizalama: %d).\n",
                capacity, CACHE_LINE_BYTES);
        free(arena);
        return NULL;
    }

    /* Zero initialize for safety */
    memset(arena->buffer, 0, capacity);

    return arena;
}

/* ══════════════════════════════════════════════════════════════
 *  Arena Alloc — bump allocator with 64-byte alignment
 * ══════════════════════════════════════════════════════════════ */

void* arena_alloc(MemoryArena* arena, size_t size) {
    if (!arena || size == 0) return NULL;

    /* Round up to 64-byte boundary to prevent cache-line splits */
    size_t aligned_size = (size + CACHE_LINE_BYTES - 1) & ~((size_t)CACHE_LINE_BYTES - 1);

    if (arena->offset + aligned_size > arena->capacity) {
        fprintf(stderr, "[Arena] HATA: Kapasite aşıldı! İstenen: %zu, Kalan: %zu, Toplam: %zu\n",
                aligned_size, arena->capacity - arena->offset, arena->capacity);
        return NULL;
    }

    void* ptr = &arena->buffer[arena->offset];
    arena->offset += aligned_size;
    arena->alloc_count++;

    if (arena->offset > arena->peak_offset) {
        arena->peak_offset = arena->offset;
    }

    return ptr;
}

/* ══════════════════════════════════════════════════════════════
 *  Arena Reset — rewind offset, keep the buffer
 * ══════════════════════════════════════════════════════════════ */

void arena_reset(MemoryArena* arena) {
    if (arena) {
        arena->offset      = 0;
        arena->alloc_count = 0;
    }
}

/* ══════════════════════════════════════════════════════════════
 *  Arena Destroy — free everything with a single call
 * ══════════════════════════════════════════════════════════════ */

void arena_destroy(MemoryArena* arena) {
    if (arena) {
        free(arena->buffer);
        arena->buffer   = NULL;
        arena->capacity = 0;
        arena->offset   = 0;
        free(arena);
    }
}

/* ══════════════════════════════════════════════════════════════
 *  Arena Stats — diagnostic output
 * ══════════════════════════════════════════════════════════════ */

void arena_print_stats(const MemoryArena* arena) {
    if (!arena) return;

    double used_mb = (double)arena->offset / (1024.0 * 1024.0);
    double cap_mb  = (double)arena->capacity / (1024.0 * 1024.0);
    double pct     = (double)arena->offset / (double)arena->capacity * 100.0;

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║         Arena Bellek İstatistikleri          ║\n");
    printf("╠══════════════════════════════════════════════╣\n");
    printf("║  Toplam Kapasite : %10.2f MB              ║\n", cap_mb);
    printf("║  Kullanılan      : %10.2f MB (%5.1f%%)     ║\n", used_mb, pct);
    printf("║  Boş Alan        : %10.2f MB              ║\n", cap_mb - used_mb);
    printf("║  Tahsis Sayısı   : %10u                  ║\n", arena->alloc_count);
    printf("║  Zirve Kullanım  : %10.2f MB              ║\n",
           (double)arena->peak_offset / (1024.0 * 1024.0));
    printf("║  Hizalama        : %10d byte             ║\n", CACHE_LINE_BYTES);
    printf("╚══════════════════════════════════════════════╝\n");
}
