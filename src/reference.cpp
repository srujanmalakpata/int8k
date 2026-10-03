#include "int8k/reference.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "kernels.hpp"

namespace int8k {
namespace {

using DotF32Fn = float (*)(const float*, const float*, std::size_t);

DotF32Fn select_f32(Backend b) {
#if INT8K_X86
  if (resolve_backend(b) == Backend::Avx2) return detail::dot_f32_avx2;
#else
  (void)resolve_backend(b);  // still validates the request
#endif
  return detail::dot_f32_scalar;
}

QuantizedReference dot_rows(std::span<const float> w_hat, std::size_t rows, std::size_t cols,
                            const QuantizedActivations& x) {
  if (x.length != cols) {
    throw std::invalid_argument("gemv_dequantized_f64: activation length must equal weight cols");
  }
  const std::vector<float> x_hat = dequantize(x);
  QuantizedReference out{std::vector<double>(rows, 0.0), std::vector<double>(rows, 0.0)};
  for (std::size_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    double abs_acc = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      // A product of two floats is exact in double (24 + 24 significant bits < 53).
      const double t = static_cast<double>(w_hat[r * cols + c]) * static_cast<double>(x_hat[c]);
      acc += t;
      abs_acc += std::fabs(t);
    }
    out.y[r] = acc;
    out.abs_sum[r] = abs_acc;
  }
  return out;
}

}  // namespace

void gemv_f32(MatrixView w, std::span<const float> x, std::span<float> y, Backend backend) {
  if (x.size() != w.cols() || y.size() != w.rows()) {
    throw std::invalid_argument("gemv_f32: shape mismatch");
  }
  const DotF32Fn dot = select_f32(backend);
  for (std::size_t r = 0; r < w.rows(); ++r) {
    y[r] = dot(w.row(r).data(), x.data(), w.cols());
  }
}

void gemm_f32(MatrixView w, MatrixView x, std::span<float> y, Backend backend) {
  if (x.cols() != w.cols() || !size_matches(y.size(), x.rows(), w.rows())) {
    throw std::invalid_argument("gemm_f32: shape mismatch");
  }
  for (std::size_t m = 0; m < x.rows(); ++m) {
    gemv_f32(w, x.row(m), y.subspan(m * w.rows(), w.rows()), backend);
  }
}

QuantizedReference gemv_dequantized_f64(const QuantizedI8Matrix& w, const QuantizedActivations& x) {
  return dot_rows(dequantize(w), w.rows, w.cols, x);
}

QuantizedReference gemv_dequantized_f64(const QuantizedQ4Matrix& w, const QuantizedActivations& x) {
  return dot_rows(dequantize(w), w.rows, w.cols, x);
}

double kernel_tolerance(std::size_t blocks, double abs_sum) {
  constexpr double u = 0x1.0p-24;  // unit roundoff of float32
  // Largest |int32 block dot|: 32 * 127 * 127 (INT8 x INT8; INT4 x INT8 is smaller).
  constexpr double kMaxBlockDot = 32.0 * 127.0 * 127.0;
  const double nb = static_cast<double>(blocks);
  const double n = nb + 8.0;
  const double gamma = n * u / (1.0 - n * u);
  // Gradual underflow: a float operation whose result is subnormal has an ABSOLUTE error of up
  // to denorm_min / 2 instead of a relative one. The worst case is the per-block scale product
  // ws * xs[b] (subnormal once it drops below FLT_MIN, about 1.2e-38), whose error is then
  // multiplied by the exact block dot (up to kMaxBlockDot). Every other operation on a term (the
  // multiply or FMA, the accumulation, the horizontal sum) adds at most one more denorm_min / 2.
  const double eta = 0.5 * static_cast<double>(std::numeric_limits<float>::denorm_min());
  const double underflow = (1.0 + gamma) * eta * (nb * (kMaxBlockDot + 4.0) + 8.0);
  return gamma * abs_sum + underflow;
}

}  // namespace int8k
