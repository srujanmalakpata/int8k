// Quantized storage formats and (de)quantization routines.
//
// Two weight formats are provided:
//   * QuantizedI8Matrix: symmetric per-row ("per-output-channel") INT8.
//   * QuantizedQ4Matrix: symmetric block-wise INT4 with one float scale per 32 elements
//     (similar in spirit to llama.cpp's Q4_0, see DESIGN.md for the differences).
// Activations (the vector side of GEMV/GEMM) are quantized on the fly into
// QuantizedActivations: symmetric INT8 with one float scale per 32-element block.
//
// NaN policy: every quantize_* function rejects non-finite input (NaN, +Inf, -Inf) by
// throwing std::invalid_argument. A single NaN would otherwise poison a whole row/block scale.
//
// The structs are plain aggregates so they are easy to inspect and serialize. Build them with
// quantize_*; gemv/gemm check every buffer size of a hand-built value, but they do not scan
// the codes, so a hand-built INT8 code of -128 (outside [-127, 127]) is a precondition
// violation that gives wrong results on the AVX2 path (see DESIGN.md, decision 1).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "int8k/matrix.hpp"

namespace int8k {

/// Number of elements that share one scale in block-wise formats. Rows of every quantized
/// format are zero-padded up to a multiple of this, so kernels never need a scalar tail loop.
inline constexpr std::size_t kBlockSize = 32;

/// Largest representable magnitude for symmetric INT8 / INT4 codes. We use the symmetric
/// ranges [-127, 127] and [-7, 7] so that negation is exact and the error bound is simple.
inline constexpr int kI8Max = 127;
inline constexpr int kQ4Max = 7;

[[nodiscard]] constexpr std::size_t blocks_for(std::size_t n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

/// Symmetric per-row INT8 weights: w[r][c] ~= scales[r] * values[r * row_stride + c].
struct QuantizedI8Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t row_stride = 0;       ///< cols rounded up to kBlockSize; padding is zero.
  std::vector<std::int8_t> values;  ///< rows * row_stride codes in [-127, 127].
  std::vector<float> scales;        ///< One scale per row: max|w[r][:]| / 127.

  [[nodiscard]] std::span<const std::int8_t> row(std::size_t r) const {
    return std::span<const std::int8_t>(values).subspan(r * row_stride, row_stride);
  }
  /// Bytes of quantized payload (codes + scales), excluding the struct itself.
  [[nodiscard]] std::size_t storage_bytes() const {
    return values.size() * sizeof(std::int8_t) + scales.size() * sizeof(float);
  }
};

/// Block-wise INT4 weights. Each 32-element block is stored as 16 bytes: byte j holds element j
/// in its low nibble and element j + 16 in its high nibble, both biased by +8 (so 0..15).
/// w[r][b * 32 + j] ~= scales[r * blocks_per_row + b] * (nibble - 8).
struct QuantizedQ4Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t blocks_per_row = 0;
  std::vector<std::uint8_t> packed;  ///< rows * blocks_per_row * 16 bytes.
  std::vector<float> scales;         ///< rows * blocks_per_row scales: max|block| / 7.

  static constexpr std::size_t kBytesPerBlock = kBlockSize / 2;

  [[nodiscard]] std::span<const std::uint8_t> row_packed(std::size_t r) const {
    return std::span<const std::uint8_t>(packed).subspan(r * blocks_per_row * kBytesPerBlock,
                                                         blocks_per_row * kBytesPerBlock);
  }
  [[nodiscard]] std::span<const float> row_scales(std::size_t r) const {
    return std::span<const float>(scales).subspan(r * blocks_per_row, blocks_per_row);
  }
  [[nodiscard]] std::size_t storage_bytes() const {
    return packed.size() + scales.size() * sizeof(float);
  }
};

/// Block-wise INT8 activations: x[b * 32 + j] ~= scales[b] * values[b * 32 + j].
struct QuantizedActivations {
  std::size_t length = 0;           ///< Logical length (number of real elements).
  std::vector<std::int8_t> values;  ///< blocks_for(length) * 32 codes, zero-padded.
  std::vector<float> scales;        ///< One scale per block: max|block| / 127.

  [[nodiscard]] std::size_t blocks() const { return scales.size(); }
};

/// Quantizes each row of `w` with its own scale (max|row| / 127).
/// Throws std::invalid_argument if `w` contains NaN or Inf.
[[nodiscard]] QuantizedI8Matrix quantize_i8(MatrixView w);

/// Quantizes `w` in 32-element blocks along each row (scale = max|block| / 7).
/// Throws std::invalid_argument if `w` contains NaN or Inf.
[[nodiscard]] QuantizedQ4Matrix quantize_q4(MatrixView w);

/// Quantizes a vector in 32-element blocks (scale = max|block| / 127).
/// Throws std::invalid_argument if `x` contains NaN or Inf.
[[nodiscard]] QuantizedActivations quantize_activations(std::span<const float> x);

/// Writes the dequantized matrix (rows * cols floats, row-major) into `out`.
void dequantize(const QuantizedI8Matrix& q, std::span<float> out);
void dequantize(const QuantizedQ4Matrix& q, std::span<float> out);
void dequantize(const QuantizedActivations& q, std::span<float> out);

/// Convenience overloads that allocate the output.
[[nodiscard]] std::vector<float> dequantize(const QuantizedI8Matrix& q);
[[nodiscard]] std::vector<float> dequantize(const QuantizedQ4Matrix& q);
[[nodiscard]] std::vector<float> dequantize(const QuantizedActivations& q);

}  // namespace int8k
