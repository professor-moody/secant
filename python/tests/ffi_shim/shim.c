/* Reference implementation of the FIXED secant C ABI v1, used ONLY to test the
 * Python ctypes plumbing (argument order, sizes, pointers, status codes) when
 * libsecant is not built. It is not libsecant and proves nothing about it.
 * Q8_0 here keeps the block scale in float32 (no fp16 rounding). */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

const char *secant_version_v1(void) { return "shim-0"; }

static int orthonormalize(const float *dirs, size_t k, size_t d, double *q) {
  for (size_t i = 0; i < k; i++) {
    double *u = q + i * d;
    double n0 = 0;
    for (size_t c = 0; c < d; c++) {
      u[c] = dirs[i * d + c];
      if (!isfinite(u[c])) return 2;
      n0 += u[c] * u[c];
    }
    n0 = sqrt(n0);
    if (!(n0 > 0)) return 2;
    for (int pass = 0; pass < 2; pass++)
      for (size_t j = 0; j < i; j++) {
        double p = 0;
        for (size_t c = 0; c < d; c++) p += u[c] * q[j * d + c];
        for (size_t c = 0; c < d; c++) u[c] -= p * q[j * d + c];
      }
    double n1 = 0;
    for (size_t c = 0; c < d; c++) n1 += u[c] * u[c];
    n1 = sqrt(n1);
    if (!(n1 >= 1e-9 * n0)) return 3;
    for (size_t c = 0; c < d; c++) u[c] /= n1;
  }
  return 0;
}

int32_t secant_project_then_heal_f32_v1(const float *directions, size_t k, size_t d_out, float *b, size_t r, double *max_leak_out) {
  if (!directions || !b || !max_leak_out || k == 0 || d_out == 0 || r == 0 || k > d_out) return 1;
  double *q = malloc(k * d_out * sizeof(double));
  double *col = malloc(d_out * sizeof(double));
  if (!q || !col) { free(q); free(col); return 4; }
  int rc = orthonormalize(directions, k, d_out, q);
  if (rc) { free(q); free(col); return rc; }
  double worst = 0;
  for (size_t j = 0; j < r; j++) {
    for (size_t i = 0; i < d_out; i++) col[i] = b[i * r + j];
    for (int pass = 0; pass < 2; pass++)
      for (size_t e = 0; e < k; e++) {
        double p = 0;
        for (size_t i = 0; i < d_out; i++) p += col[i] * q[e * d_out + i];
        for (size_t i = 0; i < d_out; i++) col[i] -= p * q[e * d_out + i];
      }
    for (size_t i = 0; i < d_out; i++) b[i * r + j] = (float)col[i];
  }
  for (size_t j = 0; j < r; j++)
    for (size_t e = 0; e < k; e++) {
      double p = 0;
      for (size_t i = 0; i < d_out; i++) p += (double)b[i * r + j] * q[e * d_out + i];
      if (fabs(p) > worst) worst = fabs(p);
    }
  *max_leak_out = worst;
  free(q); free(col);
  return 0;
}

int32_t secant_q8_0_commutation_witness_v1(const float *directions, size_t k, size_t d_out, const float *w, size_t d_in, double *out_l2, double *out_max_abs) {
  if (!directions || !w || !out_l2 || !out_max_abs || k == 0 || d_out == 0 || d_in == 0 || d_in % 32 != 0) return 1;
  double *err = malloc(d_out * d_in * sizeof(double));
  if (!err) return 4;
  for (size_t row = 0; row < d_out; row++)
    for (size_t blk = 0; blk < d_in / 32; blk++) {
      const float *x = w + row * d_in + blk * 32;
      float amax = 0;
      for (int i = 0; i < 32; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
      float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
      for (int i = 0; i < 32; i++) err[row * d_in + blk * 32 + i] = (double)(roundf(x[i] * id) * d) - (double)x[i];
    }
  double s2 = 0, mx = 0;
  for (size_t e = 0; e < k; e++)
    for (size_t c = 0; c < d_in; c++) {
      double p = 0;
      for (size_t i = 0; i < d_out; i++) p += (double)directions[e * d_out + i] * err[i * d_in + c];
      s2 += p * p;
      if (fabs(p) > mx) mx = fabs(p);
    }
  *out_l2 = sqrt(s2); *out_max_abs = mx;
  free(err);
  return 0;
}
