/* Reference driver for Secant's Q8_0 / Q4_K kernels. Links the UNMODIFIED
 * ggml-quants.c of a llama.cpp checkout.
 *
 *   ggml_ref quant   <q8_0|q4_k> <in.f32> <out.bin>    quantize (little-endian f32 input)
 *   ggml_ref dequant <q8_0|q4_k> <in.bin> <out.f32> <n> dequantize n elements
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml-common.h"
#include "ggml-quants.h"

static void *rd(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *b = malloc(sz ? sz : 1);
    if (fread(b, 1, sz, f) != (size_t)sz) { perror("fread"); exit(2); }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static void wr(const char *p, const void *b, size_t n) {
    FILE *f = fopen(p, "wb");
    if (!f) { perror(p); exit(2); }
    if (fwrite(b, 1, n, f) != n) { perror("fwrite"); exit(2); }
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage\n"); return 2; }
    int q8 = !strcmp(argv[2], "q8_0");
    if (!strcmp(argv[1], "quant")) {
        size_t nb;
        float *x = rd(argv[3], &nb);
        int64_t n = (int64_t)(nb / 4);
        if (q8) {
            block_q8_0 *y = malloc((n / QK8_0) * sizeof(block_q8_0));
            quantize_row_q8_0_ref(x, y, n);
            wr(argv[4], y, (n / QK8_0) * sizeof(block_q8_0));
        } else {
            block_q4_K *y = malloc((n / QK_K) * sizeof(block_q4_K));
            quantize_row_q4_K_ref(x, y, n);
            wr(argv[4], y, (n / QK_K) * sizeof(block_q4_K));
        }
        return 0;
    }
    if (!strcmp(argv[1], "dequant") && argc >= 6) {
        size_t nb;
        void *b = rd(argv[3], &nb);
        int64_t n = atoll(argv[5]);
        float *y = malloc(n * sizeof(float));
        if (q8) dequantize_row_q8_0(b, y, n); else dequantize_row_q4_K(b, y, n);
        wr(argv[4], y, n * sizeof(float));
        return 0;
    }
    return 2;
}
