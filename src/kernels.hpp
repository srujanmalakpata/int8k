// Internal kernel interface (not installed / not part of the public API).
//
// Each kernel computes the dot product of ONE weight row with ONE quantized activation vector,
// both laid out as `blocks` consecutive 32-element blocks. GEMV/GEMM call these per row; the
// row split and threading live in gemv.cpp, so every backend shares the same driver.
#pragma once

#include <cstddef>
#include <cstdint>

#include "int8k/cpu.hpp"

// INT8K_FORCE_PORTABLE (CMake: -DINT8K_PORTABLE=ON) compiles the x86 kernels out, so the
// portable fallback path that ARM/Apple-silicon builds take can be tested on an x86 machine.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && \
    !defined(INT8K_FORCE_PORTABLE)
#define INT8K_X86 1
#else
#define INT8K_X86 0
#endif

namespace int8k::detail {

/// Returns sum_b (ws * xs[b]) * dot(w[b], x[b]) where each dot is an exact int32 over 32 int8
/// codes and `ws` is the row's weight scale. The row scale is folded into every block's scale
/// (as for INT4) rather than applied once at the end: with large activations and small weights,
/// sum_b xs[b] * dot alone can overflow float even though the true result is finite.
using DotI8Fn = float (*)(const std::int8_t* w, float ws, const std::int8_t* x, const float* xs,
                          std::size_t blocks);

/// Returns sum_b ws[b] * xs[b] * dot(unpack(wq[b]), x[b]).
using DotQ4Fn = float (*)(const std::uint8_t* wq, const float* ws, const std::int8_t* x,
                          const float* xs, std::size_t blocks);

/// Plain float32 dot product of length n (used by the float32 baseline GEMV).
float dot_f32_scalar(const float* w, const float* x, std::size_t n);

float dot_i8_scalar(const std::int8_t* w, float ws, const std::int8_t* x, const float* xs,
                    std::size_t blocks);
float dot_q4_scalar(const std::uint8_t* wq, const float* ws, const std::int8_t* x, const float* xs,
                    std::size_t blocks);

#if INT8K_X86
// Defined in kernels_avx2.cpp with __attribute__((target("avx2,fma"))). Only call these after
// cpu_has_avx2() returned true.
float dot_f32_avx2(const float* w, const float* x, std::size_t n);
float dot_i8_avx2(const std::int8_t* w, float ws, const std::int8_t* x, const float* xs,
                  std::size_t blocks);
float dot_q4_avx2(const std::uint8_t* wq, const float* ws, const std::int8_t* x, const float* xs,
                  std::size_t blocks);
#endif

/// The kernel gemv/gemm run for a backend (Auto is resolved first; throws like resolve_backend
/// for an unavailable one). Exposed so a test can assert that Backend::Avx2 really dispatches to
/// the AVX2 kernels rather than silently falling back to scalar.
DotI8Fn select_i8(Backend b);
DotQ4Fn select_q4(Backend b);

}  // namespace int8k::detail
