// Reference implementations used as correctness oracles and benchmark baselines.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "int8k/cpu.hpp"
#include "int8k/matrix.hpp"
#include "int8k/quantize.hpp"

namespace int8k {

/// y = W x in float32, single-threaded. Backend::Scalar is a strict left-to-right loop (the
/// correctness baseline); Backend::Avx2 / Auto use an 8-wide FMA kernel, which is the fair
/// same-instruction-set baseline for benchmarking the quantized kernels.
void gemv_f32(MatrixView w, std::span<const float> x, std::span<float> y,
              Backend backend = Backend::Scalar);

/// Y = X W^T in float32. X is [m x w.cols], Y is [m x w.rows].
void gemm_f32(MatrixView w, MatrixView x, std::span<float> y, Backend backend = Backend::Scalar);

/// The value a quantized GEMV approximates, computed independently of the kernels:
/// y[r] = sum_c w_hat[r][c] * x_hat[c] in double, where w_hat and x_hat come from dequantize()
/// (so the nibble layout and scale indexing are decoded by separate code). abs_sum[r] is
/// sum_c |w_hat[r][c] * x_hat[c]|, the scale for floating-point error bounds.
struct QuantizedReference {
  std::vector<double> y;
  std::vector<double> abs_sum;
};
[[nodiscard]] QuantizedReference gemv_dequantized_f64(const QuantizedI8Matrix& w,
                                                      const QuantizedActivations& x);
[[nodiscard]] QuantizedReference gemv_dequantized_f64(const QuantizedQ4Matrix& w,
                                                      const QuantizedActivations& x);

/// Largest |kernel - reference.y[r]| any backend may legitimately show for a row of `blocks`
/// 32-element blocks: gamma(blocks + 8) * abs_sum + an underflow term, with gamma(n) = n*u/(1-n*u)
/// and u = 2^-24. The integer block dots are exact, so only float32 rounding remains. A term passes
/// through at most blocks + 5 roundings in either kernel (scale product, product, accumulation
/// chain, horizontal sum) and through 2 in the reference (dequantize of w and x); DESIGN.md has
/// the derivation. The absolute underflow term covers subnormal intermediates in the kernels
/// (up to about blocks * 3.6e-40), most importantly a scale product ws * xs[b] below FLT_MIN.
///
/// Valid range: every stored scale must be a normal float or zero (group max |value| >= 127 *
/// FLT_MIN for INT8 codes, >= 7 * FLT_MIN for INT4, i.e. above about 1.5e-36), so that the
/// oracle's dequantized values are exact up to a relative rounding. Within that range, anything
/// larger is a kernel bug. Results whose scale products underflow (max|w_row| * max|x_block|
/// below about 2e-34) are still within the bound, but only in absolute terms: they can lose all
/// relative precision, down to 0.
[[nodiscard]] double kernel_tolerance(std::size_t blocks, double abs_sum);

}  // namespace int8k
