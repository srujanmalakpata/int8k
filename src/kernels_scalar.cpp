// Portable reference kernels. These are the correctness oracle for every optimized path, so
// they favour clarity over speed (the compiler may still auto-vectorize the inner loops).
#include <cstdint>

#include "int8k/quantize.hpp"
#include "kernels.hpp"

namespace int8k::detail {

float dot_f32_scalar(const float* w, const float* x, std::size_t n) {
  // Strict left-to-right float sum. Without -ffast-math the compiler may not reorder it, so this
  // loop is NOT auto-vectorized; dot_f32_avx2 is the fair same-ISA float32 baseline.
  float acc = 0.0f;
  for (std::size_t i = 0; i < n; ++i) acc += w[i] * x[i];
  return acc;
}

float dot_i8_scalar(const std::int8_t* w, float ws, const std::int8_t* x, const float* xs,
                    std::size_t blocks) {
  float acc = 0.0f;
  for (std::size_t b = 0; b < blocks; ++b) {
    // |sum| <= 32 * 127 * 127 = 516,128: fits int32 and is exactly representable in float.
    std::int32_t sum = 0;
    for (std::size_t j = 0; j < kBlockSize; ++j) {
      sum += static_cast<std::int32_t>(w[b * kBlockSize + j]) * x[b * kBlockSize + j];
    }
    acc += static_cast<float>(sum) * (ws * xs[b]);
  }
  return acc;
}

float dot_q4_scalar(const std::uint8_t* wq, const float* ws, const std::int8_t* x, const float* xs,
                    std::size_t blocks) {
  constexpr std::size_t kHalf = kBlockSize / 2;
  float acc = 0.0f;
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint8_t* bytes = wq + b * QuantizedQ4Matrix::kBytesPerBlock;
    const std::int8_t* xb = x + b * kBlockSize;
    std::int32_t sum = 0;
    for (std::size_t j = 0; j < kHalf; ++j) {
      const int lo = (bytes[j] & 0x0F) - 8;
      const int hi = (bytes[j] >> 4) - 8;
      sum += lo * xb[j] + hi * xb[j + kHalf];
    }
    acc += static_cast<float>(sum) * (ws[b] * xs[b]);
  }
  return acc;
}

}  // namespace int8k::detail
