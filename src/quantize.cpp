#include "int8k/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace int8k {
namespace {

// Largest |x| over a range; throws on any non-finite value (see NaN policy in quantize.hpp).
float checked_abs_max(std::span<const float> xs, const char* who) {
  float amax = 0.0f;
  for (const float v : xs) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument(std::string(who) + ": input contains NaN or Inf");
    }
    amax = std::max(amax, std::fabs(v));
  }
  return amax;
}

// Scale for a symmetric group whose largest magnitude is `amax`. Normally amax / max_code, but
// rounded down by a few ulps when needed so that max_code * scale never exceeds amax: for amax
// near FLT_MAX, amax / 127 can round up and dequantizing the largest code would overflow to Inf.
float symmetric_scale(float amax, int max_code) {
  const auto code = static_cast<float>(max_code);
  float scale = amax / code;
  while (scale > 0.0f && code * scale > amax) {
    scale = std::nextafter(scale, 0.0f);
  }
  return scale;
}

// Maps v to the nearest code in [-max_code, max_code] for the given scale. A zero scale means
// the whole group is zero (or so tiny that amax / max_code underflowed), so every code is 0.
// We divide rather than multiply by 1/scale: for very small scales 1/scale overflows to +Inf.
int quantize_one(float v, float scale, int max_code) {
  if (scale == 0.0f) {
    return 0;
  }
  const float q = std::nearbyint(v / scale);
  const auto limit = static_cast<float>(max_code);
  return static_cast<int>(std::clamp(q, -limit, limit));
}

void check_size(std::size_t got, std::size_t want, const char* who) {
  if (got != want) {
    throw std::invalid_argument(std::string(who) + ": output span has the wrong size");
  }
}

}  // namespace

QuantizedI8Matrix quantize_i8(MatrixView w) {
  QuantizedI8Matrix q;
  q.rows = w.rows();
  q.cols = w.cols();
  q.row_stride = blocks_for(w.cols()) * kBlockSize;
  q.values.assign(q.rows * q.row_stride, 0);
  q.scales.assign(q.rows, 0.0f);

  for (std::size_t r = 0; r < w.rows(); ++r) {
    const auto row = w.row(r);
    const float scale = symmetric_scale(checked_abs_max(row, "quantize_i8"), kI8Max);
    q.scales[r] = scale;
    std::int8_t* out = q.values.data() + r * q.row_stride;
    for (std::size_t c = 0; c < w.cols(); ++c) {
      out[c] = static_cast<std::int8_t>(quantize_one(row[c], scale, kI8Max));
    }
  }
  return q;
}

QuantizedQ4Matrix quantize_q4(MatrixView w) {
  constexpr std::size_t kHalf = kBlockSize / 2;
  QuantizedQ4Matrix q;
  q.rows = w.rows();
  q.cols = w.cols();
  q.blocks_per_row = blocks_for(w.cols());
  // A zero code is stored as nibble 8, so the padding byte for (0, 0) is 0x88.
  q.packed.assign(q.rows * q.blocks_per_row * QuantizedQ4Matrix::kBytesPerBlock, 0x88);
  q.scales.assign(q.rows * q.blocks_per_row, 0.0f);

  for (std::size_t r = 0; r < w.rows(); ++r) {
    const auto row = w.row(r);
    for (std::size_t b = 0; b < q.blocks_per_row; ++b) {
      const std::size_t begin = b * kBlockSize;
      const std::size_t len = std::min(kBlockSize, w.cols() - begin);
      const auto block = row.subspan(begin, len);
      const float scale = symmetric_scale(checked_abs_max(block, "quantize_q4"), kQ4Max);
      q.scales[r * q.blocks_per_row + b] = scale;

      int codes[kBlockSize] = {};  // padded elements stay 0
      for (std::size_t j = 0; j < len; ++j) {
        codes[j] = quantize_one(block[j], scale, kQ4Max);
      }
      std::uint8_t* out =
          q.packed.data() + (r * q.blocks_per_row + b) * QuantizedQ4Matrix::kBytesPerBlock;
      for (std::size_t j = 0; j < kHalf; ++j) {
        const auto lo = static_cast<unsigned>(codes[j] + 8);
        const auto hi = static_cast<unsigned>(codes[j + kHalf] + 8);
        out[j] = static_cast<std::uint8_t>(lo | (hi << 4));
      }
    }
  }
  return q;
}

QuantizedActivations quantize_activations(std::span<const float> x) {
  QuantizedActivations q;
  q.length = x.size();
  const std::size_t blocks = blocks_for(x.size());
  q.values.assign(blocks * kBlockSize, 0);
  q.scales.assign(blocks, 0.0f);

  for (std::size_t b = 0; b < blocks; ++b) {
    const std::size_t begin = b * kBlockSize;
    const auto block = x.subspan(begin, std::min(kBlockSize, x.size() - begin));
    const float scale = symmetric_scale(checked_abs_max(block, "quantize_activations"), kI8Max);
    q.scales[b] = scale;
    for (std::size_t j = 0; j < block.size(); ++j) {
      q.values[begin + j] = static_cast<std::int8_t>(quantize_one(block[j], scale, kI8Max));
    }
  }
  return q;
}

void dequantize(const QuantizedI8Matrix& q, std::span<float> out) {
  check_size(out.size(), q.rows * q.cols, "dequantize(QuantizedI8Matrix)");
  for (std::size_t r = 0; r < q.rows; ++r) {
    const auto codes = q.row(r);
    for (std::size_t c = 0; c < q.cols; ++c) {
      out[r * q.cols + c] = q.scales[r] * static_cast<float>(codes[c]);
    }
  }
}

void dequantize(const QuantizedQ4Matrix& q, std::span<float> out) {
  constexpr std::size_t kHalf = kBlockSize / 2;
  check_size(out.size(), q.rows * q.cols, "dequantize(QuantizedQ4Matrix)");
  for (std::size_t r = 0; r < q.rows; ++r) {
    const auto bytes = q.row_packed(r);
    const auto scales = q.row_scales(r);
    for (std::size_t c = 0; c < q.cols; ++c) {
      const std::size_t b = c / kBlockSize;
      const std::size_t j = c % kBlockSize;
      const std::uint8_t byte = bytes[b * QuantizedQ4Matrix::kBytesPerBlock + (j % kHalf)];
      const int nibble = j < kHalf ? (byte & 0x0F) : (byte >> 4);
      out[r * q.cols + c] = scales[b] * static_cast<float>(nibble - 8);
    }
  }
}

void dequantize(const QuantizedActivations& q, std::span<float> out) {
  check_size(out.size(), q.length, "dequantize(QuantizedActivations)");
  for (std::size_t i = 0; i < q.length; ++i) {
    out[i] = q.scales[i / kBlockSize] * static_cast<float>(q.values[i]);
  }
}

std::vector<float> dequantize(const QuantizedI8Matrix& q) {
  std::vector<float> out(q.rows * q.cols);
  dequantize(q, out);
  return out;
}

std::vector<float> dequantize(const QuantizedQ4Matrix& q) {
  std::vector<float> out(q.rows * q.cols);
  dequantize(q, out);
  return out;
}

std::vector<float> dequantize(const QuantizedActivations& q) {
  std::vector<float> out(q.length);
  dequantize(q, out);
  return out;
}

}  // namespace int8k
