/* Links libsecant.a from C and exercises the three entry points plus the
 * error paths. Exit 0 = all checks passed. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "secant.h"

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "CHECK failed line %d: %s\n", __LINE__, #c); return 1; } } while (0)

int main(void) {
    CHECK(strlen(secant_version_v1()) > 0);
    enum { K = 2, D = 6, R = 3 };
    float dirs[K * D] = {1, 1, 0, 0, 0, 0,   0, 1, 1, 1, 0, 0}; /* not orthonormal */
    float b[D * R];
    for (int i = 0; i < D * R; i++) b[i] = (float)(((i * 37) % 11) - 5) * 0.25f;
    double leak = -1;
    CHECK(secant_project_then_heal_f32_v1(dirs, K, D, b, R, &leak) == 0);
    CHECK(leak >= 0 && leak < 1e-6);
    /* independent leakage check: orthonormalize by hand (Gram-Schmidt) */
    double e0[D], e1[D];
    double n0 = sqrt(2.0);
    for (int i = 0; i < D; i++) e0[i] = dirs[i] / n0;
    double dp = 0; for (int i = 0; i < D; i++) dp += dirs[D + i] * e0[i];
    double n1 = 0; for (int i = 0; i < D; i++) { e1[i] = dirs[D + i] - dp * e0[i]; n1 += e1[i] * e1[i]; }
    n1 = sqrt(n1); for (int i = 0; i < D; i++) e1[i] /= n1;
    for (int c = 0; c < R; c++) {
        double s0 = 0, s1 = 0;
        for (int i = 0; i < D; i++) { s0 += e0[i] * b[i * R + c]; s1 += e1[i] * b[i * R + c]; }
        CHECK(fabs(s0) < 1e-6 && fabs(s1) < 1e-6);
    }
    /* error paths */
    CHECK(secant_project_then_heal_f32_v1(NULL, K, D, b, R, &leak) == 1);
    CHECK(secant_project_then_heal_f32_v1(dirs, 0, D, b, R, &leak) == 1);
    float dep[K * D] = {1, 0, 0, 0, 0, 0,   2, 0, 0, 0, 0, 0};
    CHECK(secant_project_then_heal_f32_v1(dep, K, D, b, R, &leak) == 3);
    CHECK(strlen(secant_last_error_v1()) > 0);
    float nanb[D * R]; memcpy(nanb, b, sizeof nanb); nanb[4] = NAN;
    CHECK(secant_project_then_heal_f32_v1(dirs, K, D, nanb, R, &leak) == 3);
    CHECK(secant_project_then_heal_f32_v1(dirs, (size_t)-1, D, b, R, &leak) == 2);

    /* Q8_0 witness */
    enum { DI = 64 };
    static float w[D * DI];
    for (int i = 0; i < D * DI; i++) w[i] = (float)sin(0.37 * i) * 0.5f;
    double l2 = -1, mx = -1;
    CHECK(secant_q8_0_commutation_witness_v1(dirs, K, D, w, DI, &l2, &mx) == 0);
    CHECK(l2 >= 0 && mx >= 0 && mx <= l2 + 1e-12);
    CHECK(secant_q8_0_commutation_witness_v1(dirs, K, D, w, 33, &l2, &mx) == 3); /* d_in not a multiple of 32 */
    printf("c_smoke ok: version %s, leak %.3e, witness l2 %.6e\n", secant_version_v1(), leak, l2);
    return 0;
}
