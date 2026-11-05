/* Symbols ggml-quants.c expects from the rest of ggml. */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ggml_abort %s:%d: ", file, line); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap); abort();
}

/* Only referenced by the multi-row quantize_* wrappers, which this driver
 * never calls; defined so the link succeeds. */
size_t ggml_row_size(int type, int64_t ne) { (void)type; (void)ne; abort(); }
size_t ggml_type_size(int type) { (void)type; abort(); }
const char *ggml_type_name(int type) { (void)type; return "?"; }
