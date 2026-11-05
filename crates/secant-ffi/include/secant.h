/* Secant C ABI (libsecant.a / libsecant.so / libsecant.dylib). Frozen _v1 signatures.
 * Return codes: 0 ok; 1 null pointer / zero extent; 2 size overflow;
 * 3 refused (NaN/Inf, dependent directions, misaligned shape); 4 caught panic. */
#ifndef SECANT_H
#define SECANT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Static NUL-terminated version string; never free. */
const char *secant_version_v1(void);

/* Message for the last non-zero return on this thread. */
const char *secant_last_error_v1(void);

/* b <- P(P b), P = I - D^T D. directions: row-major k x d_out (orthonormalized
 * internally in float64); b: row-major d_out x r, modified in place.
 * *max_leak_out (nullable) <- max |D^T b_locked| over the stored floats. */
int32_t secant_project_then_heal_f32_v1(const float *directions, size_t k, size_t d_out,
                                        float *b, size_t r, double *max_leak_out);

/* ||D^T (Dequant_Q8_0(Quant_Q8_0(w)) - w)||_2 and its max-abs entry.
 * w: row-major d_out x d_in (d_in % 32 == 0); directions: row-major k x d_out. */
int32_t secant_q8_0_commutation_witness_v1(const float *directions, size_t k, size_t d_out,
                                           const float *w, size_t d_in,
                                           double *out_l2, double *out_max_abs);

#ifdef __cplusplus
}
#endif
#endif
