// Shared helpers for the test suite: high-precision references and analytic error bounds.
#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "int8k/int8k.hpp"

namespace int8k::testing {

/// Slightly more than half a quantization step: x / scale is itself rounded to float, so the
/// reconstruction error can exceed scale / 2 by a few ulps of 127.
inline constexpr double kHalfStep = 0.5 + 1e-5;

/// Exact-ish y = W x computed in double.
inline std::vector<double> gemv_f64(std::span<const float> w, std::size_t rows, std::size_t cols,
                                    std::span<const float> x) {
  std::vector<double> y(rows, 0.0);
  for (std::size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (std::size_t c = 0; c < cols; ++c) acc += double(w[r * cols + c]) * double(x[c]);
    y[r] = acc;
  }
  return y;
}

/// Per-element weight quantization error bound (half a step), expanded to rows x cols.
inline std::vector<double> weight_error_bound(const QuantizedI8Matrix& q) {
  std::vector<double> e(q.rows * q.cols);
  for (std::size_t r = 0; r < q.rows; ++r)
    for (std::size_t c = 0; c < q.cols; ++c) e[r * q.cols + c] = kHalfStep * q.scales[r];
  return e;
}
inline std::vector<double> weight_error_bound(const QuantizedQ4Matrix& q) {
  std::vector<double> e(q.rows * q.cols);
  for (std::size_t r = 0; r < q.rows; ++r)
    for (std::size_t c = 0; c < q.cols; ++c)
      e[r * q.cols + c] = kHalfStep * q.scales[r * q.blocks_per_row + c / kBlockSize];
  return e;
}

/// Number of 32-element blocks in a row of the quantized matrix.
inline std::size_t row_blocks(const QuantizedI8Matrix& q) {
  return q.row_stride / kBlockSize;
}
inline std::size_t row_blocks(const QuantizedQ4Matrix& q) {
  return q.blocks_per_row;
}

/// Bounds for y_quant - y_exact (the float64 product of the ORIGINAL w and x), one per row:
///   |sum(w^ x^) - sum(w x)| <= sum(|dw||x| + |w||dx| + |dw||dx|)      (quantization)
///                             + kernel_tolerance(blocks, sum|w^ x^|)    (float32 rounding)
/// where |dw| <= scale_w / 2 and |dx| <= scale_x / 2 elementwise. The worst-case bound assumes
/// all rounding errors have the same sign. It checks quantization accuracy but is too loose
/// to detect kernel bugs; exact_oracle supplies the stricter check.
template <class QMatrix>
std::vector<double> gemv_error_bound(const QMatrix& qw, std::span<const float> w,
                                     std::span<const float> x) {
  const auto ew = weight_error_bound(qw);
  const auto qx = quantize_activations(x);
  const auto ref = gemv_dequantized_f64(qw, qx);
  std::vector<double> bound(qw.rows, 0.0);
  for (std::size_t r = 0; r < qw.rows; ++r) {
    double b = 0.0;
    for (std::size_t c = 0; c < qw.cols; ++c) {
      const std::size_t i = r * qw.cols + c;
      const double ex = kHalfStep * qx.scales[c / kBlockSize];
      b += ew[i] * std::fabs(double(x[c])) + std::fabs(double(w[i])) * ex + ew[i] * ex;
    }
    bound[r] = b + kernel_tolerance(row_blocks(qw), ref.abs_sum[r]);
  }
  return bound;
}

/// The exact oracle: what the kernel must compute up to float32 rounding, sum(w^ x^) in double
/// from dequantize(), with the derived rounding tolerance per row. A dropped block, a wrong scale
/// index or a nibble-order bug shared by every backend changes the result by far more than this.
struct Oracle {
  std::vector<double> y;
  std::vector<double> tol;
};
template <class QMatrix>
Oracle exact_oracle(const QMatrix& qw, std::span<const float> x) {
  const auto ref = gemv_dequantized_f64(qw, quantize_activations(x));
  Oracle o{ref.y, std::vector<double>(qw.rows)};
  for (std::size_t r = 0; r < qw.rows; ++r)
    o.tol[r] = kernel_tolerance(row_blocks(qw), ref.abs_sum[r]);
  return o;
}

/// Tolerance for comparing two backends that do identical integer math but round float partial
/// sums in a different order: each is within kernel_tolerance of the oracle, so twice that.
template <class QMatrix>
std::vector<double> reorder_tolerance(const QMatrix& qw, std::span<const float> x) {
  auto tol = exact_oracle(qw, x).tol;
  for (auto& t : tol) t *= 2.0;
  return tol;
}

}  // namespace int8k::testing
