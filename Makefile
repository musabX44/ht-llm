# ═════════════════════════════════════════════════════════════════
#  HT-LLM: Hierarchical Tree Language Model — Makefile
# ═════════════════════════════════════════════════════════════════

CC       := gcc
TARGET   := ht_llm
SRCS     := main.c arena.c context.c embedding.c micro_expert.c \
            model.c simd_math.c tokenizer.c tree.c train.c checkpoint.c
OBJS     := $(SRCS:.c=.o)

# GCC auto-parallelizes independent loops in training code. THREADS is the
# compile-time ceiling; OMP_NUM_THREADS controls the runtime worker count.
THREADS  ?= 4
CFLAGS   := -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -O2 -mavx2 -mfma \
            -fopenmp -ftree-parallelize-loops=$(THREADS)
LDFLAGS  := -lm -fopenmp

# Debug: symbols + AddressSanitizer, optimization and auto-parallelization off
DEBUG_CFLAGS := -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -g -O0 \
                -fsanitize=address -fno-omit-frame-pointer -fopenmp

.PHONY: all debug run clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c ht_llm.h simd_math.h
	$(CC) $(CFLAGS) -c $< -o $@

debug:
	$(CC) $(DEBUG_CFLAGS) -o $(TARGET)_dbg $(SRCS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET)_dbg
