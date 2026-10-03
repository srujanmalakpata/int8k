#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "int8k/int8k.hpp"
#include "test_helpers.hpp"

using namespace int8k;
using namespace int8k::testing;

namespace {

std::vector<Backend> available_backends() {
  std::vector<Backend> out = {Backend::Scalar};
  if (cpu_has_avx2()) out.push_back(Backend::Avx2);
  return out;
}

}  // namespace

// Integer-valued inputs whose row/block maxima are exactly the top code make quantization
// lossless, so the quantized GEMV must reproduce the exact integer result on every backend.
TEST(Gemv, ExactWhenQuantizationIsLossless) {
  const std::size_t rows = 3, cols = 64;
  std::vector<float> w(rows * cols), x(cols);
  for (std::size_t c = 0; c < cols; ++c) {
    w[0 * cols + c] = static_cast<float>(int(c % 15) - 7);  // in [-7, 7]
    w[1 * cols + c] = (c % 2 == 0) ? 7.0f : -7.0f;
    w[2 * cols + c] = 0.0f;
    x[c] = static_cast<float>(int(c % 255) - 127);
  }
  x[0] = 127.0f;
  x[32] = -127.0f;  // each activation block hits +/-127 so its scale is exactly 1
  for (std::size_t c = 0; c < cols; c += 15) w[c] = 7.0f;  // each Q4 block of row 0 has a 7

  std::vector<double> expected(rows, 0.0);
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) expected[r] += double(w[r * cols + c]) * x[c];

  // Q4 block scales are exactly 7 / 7 = 1 here. (The INT8 per-row scale would be 7 / 127,
  // which is not exactly representable, so INT8 is not lossless for this input; see the INT8
  // variant below.)
  const auto q4 = quantize_q4(MatrixView(w, rows, cols));
  for (Backend b : available_backends()) {
    std::vector<float> y(rows);
    gemv(q4, x, y, {.backend = b});
    for (std::size_t r = 0; r < rows; ++r) EXPECT_EQ(double(y[r]), expected[r]) << backend_name(b);
  }
}

// INT8 is lossless when every weight is an integer in [-127, 127] and each row contains +/-127
// (row scale exactly 127 / 127 = 1), and every activation block contains +/-127.
TEST(Gemv, Int8ExactWhenQuantizationIsLossless) {
  const std::size_t rows = 4, cols = 96;
  std::vector<float> w(rows * cols), x(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      w[r * cols + c] = static_cast<float>(int((c * 37 + r * 11) % 255) - 127);  // [-127, 127]
    }
    w[r * cols + r] = (r % 2 == 0) ? 127.0f : -127.0f;
  }
  for (std::size_t c = 0; c < cols; ++c) x[c] = static_cast<float>(int((c * 53) % 255) - 127);
  for (std::size_t b = 0; b < cols / kBlockSize; ++b) x[b * kBlockSize + 5] = -127.0f;

  // |y| <= 96 * 127 * 127 < 2^24, so the exact integer result is representable in float.
  const auto expected = gemv_f64(w, rows, cols, x);
  const auto qi = quantize_i8(MatrixView(w, rows, cols));
  for (float s : qi.scales) ASSERT_EQ(s, 1.0f);
  for (Backend b : available_backends()) {
    std::vector<float> y(rows);
    gemv(qi, x, y, {.backend = b});
    for (std::size_t r = 0; r < rows; ++r) EXPECT_EQ(double(y[r]), expected[r]) << backend_name(b);
  }
}

// Every backend must agree with the exact oracle sum(w^ x^) to within float32 rounding, at
// LLM widths. Unlike the loose quantization bound, this catches a single dropped block.
TEST(Gemv, MatchesExactOracleAtLlmWidths) {
  for (std::size_t cols : {4096u, 11008u}) {
    const std::size_t rows = 16;
    const auto w = random_normal(rows * cols, cols, 0.02f);
    const auto x = random_normal(cols, cols + 1);
    const auto qi = quantize_i8(MatrixView(w, rows, cols));
    const auto q4 = quantize_q4(MatrixView(w, rows, cols));
    const auto oi = exact_oracle(qi, x);
    const auto o4 = exact_oracle(q4, x);
    for (Backend b : available_backends()) {
      std::vector<float> yi(rows), y4(rows);
      gemv(qi, x, yi, {.backend = b});
      gemv(q4, x, y4, {.backend = b});
      for (std::size_t r = 0; r < rows; ++r) {
        EXPECT_NEAR(yi[r], oi.y[r], oi.tol[r]) << backend_name(b) << " int8 K=" << cols;
        EXPECT_NEAR(y4[r], o4.y[r], o4.tol[r]) << backend_name(b) << " q4 K=" << cols;
      }
    }
  }
}

// Large but finite results must stay finite. Accumulating sum_b xs[b] * dot before applying the
// row scale overflows (1e37 / 127 * 516,128) and returns inf although the true result is 6.4e35.
// Folding the row scale into each block scale avoids that overflow.
TEST(Gemv, LargeFiniteResultsDoNotOverflow) {
  const std::size_t rows = 2, cols = 64;
  for (const auto& [wv, xv] : {std::pair{1e-3f, 1e37f}, std::pair{1e37f, 1e-3f}}) {
    const std::vector<float> w(rows * cols, wv);
    const std::vector<float> x(cols, xv);
    const double expected = double(cols) * double(wv) * double(xv);
    const auto qi = quantize_i8(MatrixView(w, rows, cols));
    const auto q4 = quantize_q4(MatrixView(w, rows, cols));
    for (Backend b : available_backends()) {
      std::vector<float> yi(rows), y4(rows), gi(rows), g4(rows);
      gemv(qi, x, yi, {.backend = b});
      gemv(q4, x, y4, {.backend = b});
      gemm(qi, MatrixView(x, 1, cols), gi, {.backend = b});
      gemm(q4, MatrixView(x, 1, cols), g4, {.backend = b});
      for (const auto* y : {&yi, &y4, &gi, &g4}) {
        for (float v : *y) {
          ASSERT_TRUE(std::isfinite(v)) << backend_name(b) << " w=" << wv << " x=" << xv;
          EXPECT_NEAR(double(v), expected, 1e-5 * expected) << backend_name(b);
        }
      }
    }
  }
}

TEST(Gemv, MatchesFloatReferenceWithinDerivedBound) {
  const std::size_t rows = 96, cols = 1000;
  const auto w = random_normal(rows * cols, 10, 0.02f);
  const auto x = random_normal(cols, 11);
  const auto exact = gemv_f64(w, rows, cols, x);
  const auto qi = quantize_i8(MatrixView(w, rows, cols));
  const auto q4 = quantize_q4(MatrixView(w, rows, cols));
  const auto bound_i8 = gemv_error_bound(qi, w, x);
  const auto bound_q4 = gemv_error_bound(q4, w, x);

  for (Backend b : available_backends()) {
    std::vector<float> yi(rows), y4(rows);
    gemv(qi, x, yi, {.backend = b});
    gemv(q4, x, y4, {.backend = b});
    for (std::size_t r = 0; r < rows; ++r) {
      EXPECT_LE(std::fabs(yi[r] - exact[r]), bound_i8[r]) << backend_name(b) << " int8 row " << r;
      EXPECT_LE(std::fabs(y4[r] - exact[r]), bound_q4[r]) << backend_name(b) << " q4 row " << r;
    }
    // Sanity: int8 should be much more accurate than int4 on Gaussian data.
    const auto ei = compare(std::vector<float>(exact.begin(), exact.end()), yi);
    const auto e4 = compare(std::vector<float>(exact.begin(), exact.end()), y4);
    EXPECT_GT(ei.snr_db, 30.0);
    EXPECT_GT(e4.snr_db, 12.0);
    EXPECT_GT(ei.snr_db, e4.snr_db + 10.0);
  }
}

TEST(Gemv, Avx2MatchesScalar) {
  if (!cpu_has_avx2()) GTEST_SKIP() << "AVX2 not available on this CPU";
  const std::size_t rows = 50, cols = 4096 + 17;
  const auto w = random_normal(rows * cols, 20);
  const auto x = random_normal(cols, 21);
  const auto qi = quantize_i8(MatrixView(w, rows, cols));
  const auto q4 = quantize_q4(MatrixView(w, rows, cols));
  std::vector<float> s(rows), v(rows);

  gemv(qi, x, s, {.backend = Backend::Scalar});
  gemv(qi, x, v, {.backend = Backend::Avx2});
  const auto tol_i8 = reorder_tolerance(qi, x);
  for (std::size_t r = 0; r < rows; ++r) EXPECT_NEAR(s[r], v[r], tol_i8[r]) << "int8 row " << r;

  gemv(q4, x, s, {.backend = Backend::Scalar});
  gemv(q4, x, v, {.backend = Backend::Avx2});
  const auto tol_q4 = reorder_tolerance(q4, x);
  for (std::size_t r = 0; r < rows; ++r) EXPECT_NEAR(s[r], v[r], tol_q4[r]) << "q4 row " << r;
}

// The row split only changes WHICH thread computes a row, never how, so results must be
// bit-identical to the single-threaded run for every thread count (including more threads
// than rows and the "0 = hardware concurrency" setting).
TEST(Gemv, ThreadedIsBitIdenticalToSingleThreaded) {
  const std::size_t rows = 257, cols = 300;
  const auto w = random_normal(rows * cols, 30);
  const auto x = random_normal(cols, 31);
  const auto qi = quantize_i8(MatrixView(w, rows, cols));
  const auto q4 = quantize_q4(MatrixView(w, rows, cols));
  for (Backend b : available_backends()) {
    std::vector<float> ref_i8(rows), ref_q4(rows);
    gemv(qi, x, ref_i8, {.backend = b, .threads = 1});
    gemv(q4, x, ref_q4, {.backend = b, .threads = 1});
    for (unsigned t : {0u, 2u, 3u, 4u, 7u, 300u}) {
      std::vector<float> yi(rows, -1.0f), y4(rows, -1.0f);
      const KernelOptions opts{.backend = b, .threads = t, .min_rows_per_thread = 1};
      gemv(qi, x, yi, opts);
      gemv(q4, x, y4, opts);
      EXPECT_EQ(yi, ref_i8) << backend_name(b) << " threads=" << t;
      EXPECT_EQ(y4, ref_q4) << backend_name(b) << " threads=" << t;
    }
  }
}

TEST(Gemv, PrequantizedActivationOverloadMatches) {
  const std::size_t rows = 40, cols = 130;
  const auto w = random_normal(rows * cols, 40);
  const auto x = random_normal(cols, 41);
  const auto qi = quantize_i8(MatrixView(w, rows, cols));
  const auto qx = quantize_activations(x);
  std::vector<float> a(rows), b(rows);
  gemv(qi, x, a);
  gemv(qi, qx, b);
  EXPECT_EQ(a, b);
}

TEST(Gemv, ZeroInputGivesZeroOutput) {
  const std::size_t rows = 8, cols = 45;
  const auto w = random_normal(rows * cols, 50);
  const std::vector<float> x(cols, 0.0f);
  const auto q4 = quantize_q4(MatrixView(w, rows, cols));
  for (Backend b : available_backends()) {
    std::vector<float> y(rows, 1.0f);
    gemv(q4, x, y, {.backend = b});
    for (float v : y) EXPECT_EQ(v, 0.0f);
  }
}

TEST(Gemv, ShapeMismatchesThrow) {
  const std::vector<float> w(4 * 10, 1.0f);
  const auto qi = quantize_i8(MatrixView(w, 4, 10));
  const auto q4 = quantize_q4(MatrixView(w, 4, 10));
  std::vector<float> x_bad(9), x(10), y_bad(3), y(4);
  EXPECT_THROW(gemv(qi, x_bad, y), std::invalid_argument);
  EXPECT_THROW(gemv(qi, x, y_bad), std::invalid_argument);
  EXPECT_THROW(gemv(q4, x_bad, y), std::invalid_argument);
  EXPECT_THROW(gemv(q4, x, y_bad), std::invalid_argument);
  EXPECT_THROW(gemv(qi, quantize_activations(x_bad), y), std::invalid_argument);
}

// The formats are public aggregates; a hand-built value with inconsistent sizes must be rejected
// before any kernel reads past a buffer; inconsistent sizes would cause a heap-buffer-overflow.
TEST(Gemv, MalformedWeightsAreRejected) {
  const std::vector<float> w(4 * 40, 1.0f);
  const std::vector<float> x(40, 1.0f);
  std::vector<float> y(4), y2(8);

  auto qi = quantize_i8(MatrixView(w, 4, 40));
  qi.values.resize(qi.row_stride);  // one row of codes for a 4-row matrix
  EXPECT_THROW(gemv(qi, x, y), std::invalid_argument);
  auto qi2 = quantize_i8(MatrixView(w, 4, 40));
  qi2.scales.pop_back();
  EXPECT_THROW(gemv(qi2, x, y), std::invalid_argument);
  auto qi3 = quantize_i8(MatrixView(w, 4, 40));
  qi3.row_stride = 40;  // not a multiple of the block size
  EXPECT_THROW(gemv(qi3, x, y), std::invalid_argument);

  auto q4 = quantize_q4(MatrixView(w, 4, 40));
  q4.packed.resize(q4.packed.size() - 1);
  EXPECT_THROW(gemv(q4, x, y), std::invalid_argument);
  auto q4b = quantize_q4(MatrixView(w, 4, 40));
  q4b.scales.resize(3);
  EXPECT_THROW(gemv(q4b, x, y), std::invalid_argument);
  auto q4c = quantize_q4(MatrixView(w, 4, 40));
  q4c.blocks_per_row = 1;
  EXPECT_THROW(gemv(q4c, x, y), std::invalid_argument);

  std::vector<float> x_data(80, 1.0f);
  EXPECT_THROW(gemm(qi, MatrixView(x_data, 2, 40), y2), std::invalid_argument);
  EXPECT_THROW(gemm(q4, MatrixView(x_data, 2, 40), y2), std::invalid_argument);
}

TEST(Gemv, NaNActivationIsRejected) {
  const std::vector<float> w(2 * 8, 1.0f);
  const auto qi = quantize_i8(MatrixView(w, 2, 8));
  std::vector<float> x(8, 1.0f), y(2);
  x[3] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(gemv(qi, x, y), std::invalid_argument);
}

// Folding the row scale into each block (ws * xs[b]) trades the overflow above for underflow:
// when max|w_row| * max|x_block| is below about 2e-34 (INT8), the scale product is subnormal or
// flushes to 0, and the result loses relative precision, possibly all of it. kernel_tolerance
// bounds this with an absolute term (see reference.hpp); this test pins that behaviour.
TEST(Gemv, UnderflowingScaleProductsStayWithinAbsoluteBound) {
  struct Case {
    float w;
    float x;
    std::size_t cols;
  };
  // 1e-20 * 8.07e-22: the INT8 scale product is about 5e-46 and rounds to 0, so INT8 returns
  // exactly 0 although the true result is 3.3e-38. 1e-19 * 1e-19: the product is subnormal
  // (about 443 * denorm_min), so its rounding error is about 0.1% instead of 6e-8.
  for (const Case& c : {Case{1e-20f, 8.07e-22f, 4096}, Case{1e-19f, 1e-19f, 64}}) {
    const std::size_t rows = 2;
    const std::vector<float> w(rows * c.cols, c.w);
    const std::vector<float> x(c.cols, c.x);
    const auto qi = quantize_i8(MatrixView(w, rows, c.cols));
    const auto q4 = quantize_q4(MatrixView(w, rows, c.cols));
    const auto oi = exact_oracle(qi, x);
    const auto o4 = exact_oracle(q4, x);
    for (Backend b : available_backends()) {
      std::vector<float> yi(rows), y4(rows);
      gemv(qi, x, yi, {.backend = b});
      gemv(q4, x, y4, {.backend = b});
      for (std::size_t r = 0; r < rows; ++r) {
        ASSERT_TRUE(std::isfinite(yi[r]) && std::isfinite(y4[r]));
        EXPECT_NEAR(yi[r], oi.y[r], oi.tol[r]) << backend_name(b) << " int8 w=" << c.w;
        EXPECT_NEAR(y4[r], o4.y[r], o4.tol[r]) << backend_name(b) << " q4 w=" << c.w;
      }
    }
  }
}

// The kernels do not check scales for NaN (only quantize_* rejects non-finite input). A NaN in a
// hand-built struct therefore propagates like in a float GEMV: a NaN weight scale poisons only
// its own row, but a NaN activation block scale poisons every output, because every row's dot
// product includes every activation block.
TEST(Gemv, NaNScalePoisonsItsRowOrEveryRow) {
  const std::size_t rows = 3, cols = 64;
  const auto w = random_normal(rows * cols, 90);
  const auto x = random_normal(cols, 91);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (Backend b : available_backends()) {
    auto qi = quantize_i8(MatrixView(w, rows, cols));
    qi.scales[1] = nan;
    auto q4 = quantize_q4(MatrixView(w, rows, cols));
    q4.scales[1 * q4.blocks_per_row + 1] = nan;  // row 1, block 1
    for (int fmt = 0; fmt < 2; ++fmt) {
      std::vector<float> y(rows);
      if (fmt == 0) gemv(qi, x, y, {.backend = b});
      if (fmt == 1) gemv(q4, x, y, {.backend = b});
      EXPECT_FALSE(std::isnan(y[0])) << backend_name(b) << " fmt " << fmt;
      EXPECT_TRUE(std::isnan(y[1])) << backend_name(b) << " fmt " << fmt;
      EXPECT_FALSE(std::isnan(y[2])) << backend_name(b) << " fmt " << fmt;
    }

    auto qx = quantize_activations(x);
    qx.scales[0] = nan;
    const auto clean_i8 = quantize_i8(MatrixView(w, rows, cols));
    std::vector<float> y(rows);
    gemv(clean_i8, qx, y, {.backend = b});
    for (float v : y) EXPECT_TRUE(std::isnan(v)) << backend_name(b);
  }
}

// Regression check: on typical LLM-like data, an all-zero output passes the worst-case
// quantization bound for most rows but fails the exact oracle unless the true result is
// within rounding tolerance of zero. Printed counts are recorded in DESIGN.md and VERIFICATION.md.
TEST(Gemv, OracleRejectsZeroOutputThatTheLooseBoundAccepts) {
  const std::size_t rows = 64;
  for (std::size_t cols : {200u, 4096u, 11008u}) {
    const auto w = random_normal(rows * cols, 7000 + cols, 0.02f);
    const auto x = random_normal(cols, 8000 + cols);
    const auto exact = gemv_f64(w, rows, cols, x);
    const auto qi = quantize_i8(MatrixView(w, rows, cols));
    const auto q4 = quantize_q4(MatrixView(w, rows, cols));
    const auto bound_i8 = gemv_error_bound(qi, w, x);
    const auto bound_q4 = gemv_error_bound(q4, w, x);
    const auto oracle_i8 = exact_oracle(qi, x);
    const auto oracle_q4 = exact_oracle(q4, x);
    std::size_t loose_i8 = 0, loose_q4 = 0, oracle_ok_i8 = 0, oracle_ok_q4 = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      loose_i8 += std::fabs(exact[r]) <= bound_i8[r] ? 1 : 0;
      loose_q4 += std::fabs(exact[r]) <= bound_q4[r] ? 1 : 0;
      oracle_ok_i8 += std::fabs(oracle_i8.y[r]) <= oracle_i8.tol[r] ? 1 : 0;
      oracle_ok_q4 += std::fabs(oracle_q4.y[r]) <= oracle_q4.tol[r] ? 1 : 0;
    }
    std::printf(
        "[probe] K=%zu: y=0 passes the loose bound for %zu/64 INT8 and %zu/64 INT4 rows, "
        "and the exact oracle for %zu/64 INT8 and %zu/64 INT4 rows\n",
        cols, loose_i8, loose_q4, oracle_ok_i8, oracle_ok_q4);
    EXPECT_LE(oracle_ok_i8, 2u) << "K=" << cols;
    EXPECT_LE(oracle_ok_q4, 2u) << "K=" << cols;
  }
}
