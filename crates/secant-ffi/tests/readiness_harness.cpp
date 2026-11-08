// Single-buffer libFuzzer readiness harness for the Secant C ABI (`secant.h`).
// Every supplied byte participates in a finite numerical FFI input. This is a
// readiness driver, not a container parser or an inference harness.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "secant.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size == 0 || size > 16384) return 0;
    constexpr size_t D = 4, R = 2, N = 32;
    float directions[D] = {1.f, 2.f, 3.f, 0.f};
    float b[D * R] = {};
    float w[D * N] = {};
    for (size_t i = 0; i < size; ++i) {
        const float v = (static_cast<int>(data[i]) - 128) / 128.f;
        b[i % (D * R)] += v;
        w[i % (D * N)] += v;
    }
    double leak = 0, l2 = 0, max_abs = 0;
    if (secant_project_then_heal_f32_v1(directions, 1, D, b, R, &leak) != 0) std::abort();
    if (size >= 32) {
        if (secant_q8_0_commutation_witness_v1(directions, 1, D, w, N, &l2, &max_abs) != 0) std::abort();
    }
    if (std::getenv("SECANT_READINESS_MARKER")) std::fputs("secant-ffi-reached\n", stderr);
    return 0;
}
