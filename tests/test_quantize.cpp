#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "int8k/int8k.hpp"
#include "test_helpers.hpp"

using namespace int8k;
using int8k::testing::kHalfStep;

namespace {

MatrixView view(const std::vector<float>& v, std::size_t rows, std::size_t cols) {
  return MatrixView(v, rows, cols);
}

}  // namespace

TEST(QuantizeI8, ScaleIsRowAbsMaxOver127) {
  const std::vector<float> w = {1.0f, -2.0f, 0.5f,  //
                                0.0f, 0.25f, -0.125f};
  const auto q = quantize_i8(view(w, 2, 3));
  ASSERT_EQ(q.scales.size(), 2u);
  EXPECT_FLOAT_EQ(q.scales[0], 2.0f / 127.0f);
  EXPECT_FLOAT_EQ(q.scales[1], 0.25f / 127.0f);
  // The largest-magnitude element of each row maps to +/-127.
  EXPECT_EQ(q.row(0)[1], -127);
  EXPECT_EQ(q.row(1)[1], 127);
}

TEST(QuantizeI8, RowsArePaddedWithZerosToBlockMultiple) {
  const std::vector<float> w(3 * 33, 1.0f);
  const auto q = quantize_i8(view(w, 3, 33));
  EXPECT_EQ(q.row_stride, 64u);
  EXPECT_EQ(q.values.size(), 3u * 64u);
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 33; c < 64; ++c) EXPECT_EQ(q.row(r)[c], 0) << r << "," << c;
  }
}

TEST(QuantizeI8, RoundTripErrorWithinHalfStep) {
  const std::size_t rows = 37, cols = 211;
  const auto w = random_normal(rows * cols, 1);
  const auto q = quantize_i8(view(w, rows, cols));
  const auto back = dequantize(q);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      const std::size_t i = r * cols + c;
      EXPECT_LE(std::fabs(back[i] - w[i]), kHalfStep * q.scales[r]) << "at " << r << "," << c;
    }
  }
}

TEST(QuantizeQ4, PackingLayoutLowNibbleFirstHalfHighNibbleSecondHalf) {
  // Block of 32 with values -7..7 so the scale is exactly 1 and codes equal the values.
  std::vector<float> w(32);
  for (std::size_t j = 0; j < 32; ++j) w[j] = static_cast<float>(int(j % 15) - 7);
  w[0] = 7.0f;  // ensure amax = 7
  const auto q = quantize_q4(view(w, 1, 32));
  ASSERT_EQ(q.packed.size(), 16u);
  EXPECT_FLOAT_EQ(q.scales[0], 1.0f);
  for (std::size_t j = 0; j < 16; ++j) {
    EXPECT_EQ((q.packed[j] & 0x0F) - 8, static_cast<int>(w[j])) << "low nibble " << j;
    EXPECT_EQ((q.packed[j] >> 4) - 8, static_cast<int>(w[j + 16])) << "high nibble " << j;
  }
  EXPECT_EQ(dequantize(q), w);  // exactly representable
}

TEST(QuantizeQ4, EachBlockHasItsOwnScale) {
  std::vector<float> w(64, 0.0f);
  w[3] = 7.0f;     // block 0: amax 7   -> scale 1
  w[40] = -0.07f;  // block 1: amax .07 -> scale .01
  const auto q = quantize_q4(view(w, 1, 64));
  ASSERT_EQ(q.blocks_per_row, 2u);
  EXPECT_FLOAT_EQ(q.scales[0], 1.0f);
  EXPECT_FLOAT_EQ(q.scales[1], 0.07f / 7.0f);
  const auto back = dequantize(q);
  EXPECT_FLOAT_EQ(back[3], 7.0f);
  EXPECT_NEAR(back[40], -0.07f, 1e-7f);  // not crushed by block 0's larger scale
}

TEST(QuantizeQ4, RoundTripErrorWithinHalfBlockStep) {
  const std::size_t rows = 19, cols = 75;  // 75 = 2 full blocks + a tail of 11
  const auto w = random_normal(rows * cols, 2);
  const auto q = quantize_q4(view(w, rows, cols));
  EXPECT_EQ(q.blocks_per_row, 3u);
  const auto back = dequantize(q);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      const std::size_t i = r * cols + c;
      const float scale = q.scales[r * q.blocks_per_row + c / kBlockSize];
      EXPECT_LE(std::fabs(back[i] - w[i]), kHalfStep * scale) << "at " << r << "," << c;
    }
  }
}

TEST(QuantizeQ4, StorageIsFiveBitsPerWeightWithFloatScales) {
  const std::vector<float> w(64 * 4096, 0.5f);
  const auto q = quantize_q4(view(w, 64, 4096));
  const double bits_per_weight = 8.0 * double(q.storage_bytes()) / double(w.size());
  EXPECT_DOUBLE_EQ(bits_per_weight, 5.0);  // 4 bits + one 32-bit float scale per 32 weights
}

TEST(QuantizeActivations, BlockScalesAndPadding) {
  std::vector<float> x(40, 0.0f);
  x[5] = -3.0f;
  x[35] = 0.5f;
  const auto q = quantize_activations(x);
  EXPECT_EQ(q.length, 40u);
  ASSERT_EQ(q.blocks(), 2u);
  EXPECT_EQ(q.values.size(), 64u);
  EXPECT_FLOAT_EQ(q.scales[0], 3.0f / 127.0f);
  EXPECT_FLOAT_EQ(q.scales[1], 0.5f / 127.0f);
  EXPECT_EQ(q.values[5], -127);
  EXPECT_EQ(q.values[35], 127);
  for (std::size_t i = 40; i < 64; ++i) EXPECT_EQ(q.values[i], 0);
}

// ---- Edge cases ------------------------------------------------------------------------------

TEST(EdgeCases, AllZerosGiveZeroScaleAndZeroCodes) {
  const std::vector<float> w(5 * 70, 0.0f);
  const auto qi = quantize_i8(view(w, 5, 70));
  const auto q4 = quantize_q4(view(w, 5, 70));
  for (float s : qi.scales) EXPECT_EQ(s, 0.0f);
  for (float s : q4.scales) EXPECT_EQ(s, 0.0f);
  EXPECT_EQ(dequantize(qi), w);
  EXPECT_EQ(dequantize(q4), w);
  const auto qa = quantize_activations(std::vector<float>(70, 0.0f));
  for (auto v : qa.values) EXPECT_EQ(v, 0);
}

TEST(EdgeCases, MaxMagnitudeValuesDoNotOverflow) {
  const float big = std::numeric_limits<float>::max();
  const std::vector<float> w = {big, -big, big / 2, 0.0f};
  const auto qi = quantize_i8(view(w, 1, 4));
  const auto back = dequantize(qi);
  for (std::size_t i = 0; i < w.size(); ++i) {
    EXPECT_TRUE(std::isfinite(back[i]));
    EXPECT_LE(std::fabs(double(back[i]) - double(w[i])), kHalfStep * double(qi.scales[0]));
  }
  const auto q4 = quantize_q4(view(w, 1, 4));
  for (float v : dequantize(q4)) EXPECT_TRUE(std::isfinite(v));
}

// For normal inputs the reconstruction error is at most scale / 2. When amax / max_code is
// subnormal, the scale itself has few significant bits and symmetric_scale rounds it down, so
// the largest values clamp to max_code and can be off by amax - max_code * scale (all of amax
// when the scale underflows to 0). The documented bound is the larger of the two.
static double group_bound(float amax, float scale, int max_code) {
  return std::max(kHalfStep * double(scale), double(amax) - max_code * double(scale));
}

TEST(EdgeCases, SubnormalInputsStayWithinDocumentedBound) {
  const float dm = std::numeric_limits<float>::denorm_min();
  const float fmin = std::numeric_limits<float>::min();
  const std::vector<std::vector<float>> inputs = {
      {dm, -dm, 0.0f, 4 * dm},
      {100 * dm, 3 * dm},             // amax / 127 rounds to 0: scale 0, everything becomes 0
      {200 * dm, -150 * dm, 7 * dm},  // scale = 1 * denorm_min: 200 clamps to 127
      {0.5f * fmin, -0.3f * fmin, 1e-3f * fmin},
      {100 * fmin, -37 * fmin, fmin},  // normal values, but amax / 127 is subnormal
  };
  for (const auto& w : inputs) {
    const std::size_t n = w.size();
    float amax = 0.0f;
    for (float v : w) amax = std::max(amax, std::fabs(v));

    const auto qi = quantize_i8(view(w, 1, n));
    const auto bi = dequantize(qi);
    const auto q4 = quantize_q4(view(w, 1, n));
    const auto b4 = dequantize(q4);
    const auto qa = quantize_activations(w);
    const auto ba = dequantize(qa);
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_FALSE(std::isnan(bi[i]) || std::isnan(b4[i]) || std::isnan(ba[i]));
      const double v = w[i];
      EXPECT_LE(std::fabs(bi[i] - v), group_bound(amax, qi.scales[0], kI8Max)) << i << " i8";
      EXPECT_LE(std::fabs(b4[i] - v), group_bound(amax, q4.scales[0], kQ4Max)) << i << " q4";
      EXPECT_LE(std::fabs(ba[i] - v), group_bound(amax, qa.scales[0], kI8Max)) << i << " act";
    }
  }
  // The case that breaks the plain scale / 2 bound: the scale underflows to 0.
  const std::vector<float> w = {100 * dm, 3 * dm};
  const auto q = quantize_i8(view(w, 1, 2));
  EXPECT_EQ(q.scales[0], 0.0f);
  EXPECT_EQ(dequantize(q), (std::vector<float>{0.0f, 0.0f}));
}

TEST(EdgeCases, NonFiniteInputIsRejected) {
  for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    std::vector<float> w(2 * 40, 1.0f);
    w[45] = bad;
    EXPECT_THROW((void)quantize_i8(view(w, 2, 40)), std::invalid_argument);
    EXPECT_THROW((void)quantize_q4(view(w, 2, 40)), std::invalid_argument);
    EXPECT_THROW((void)quantize_activations(w), std::invalid_argument);
  }
}

TEST(EdgeCases, EmptyMatrices) {
  const std::vector<float> none;
  const auto qi = quantize_i8(view(none, 0, 0));
  EXPECT_EQ(qi.values.size(), 0u);
  const auto q4 = quantize_q4(view(none, 0, 16));
  EXPECT_EQ(q4.packed.size(), 0u);
  EXPECT_EQ(quantize_activations(none).blocks(), 0u);
}

TEST(EdgeCases, MatrixViewRejectsWrongSize) {
  const std::vector<float> w(10);
  EXPECT_THROW(MatrixView(w, 3, 4), std::invalid_argument);
  const MatrixView v(w, 2, 5);
  EXPECT_EQ(v.rows(), 2u);
  EXPECT_EQ(v.cols(), 5u);
  EXPECT_EQ(v.row(1).data(), w.data() + 5);
}

// rows * cols must not wrap around: 2^32 * 2^32 is 0 in 64-bit arithmetic, which would accept an
// empty buffer for a huge matrix and let quantize_* read out of bounds.
TEST(EdgeCases, MatrixViewRejectsOverflowingShape) {
  const std::vector<float> none;
  const std::size_t big = std::size_t{1} << 32;
  EXPECT_THROW(MatrixView(none, big, big), std::invalid_argument);
  EXPECT_THROW(MatrixView(none, SIZE_MAX, 2), std::invalid_argument);
  EXPECT_NO_THROW(MatrixView(none, 0, big));
  EXPECT_NO_THROW(MatrixView(none, big, 0));
}

TEST(EdgeCases, DequantizeRejectsWrongOutputSize) {
  const std::vector<float> w(6, 1.0f);
  const auto q = quantize_i8(view(w, 2, 3));
  std::vector<float> out(5);
  EXPECT_THROW(dequantize(q, out), std::invalid_argument);
}
