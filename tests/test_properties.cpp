// Property-style randomized tests: hand-picked boundary widths, LLM layer widths and random
// shapes (deliberately including sizes that are not multiples of the 32-element block, and tiny
// sizes like 1x1), random value scales, both weight formats, every available backend and a
// threaded run. Every case checks:
//   1. every backend vs the exact oracle sum(w^ x^) (from dequantize(), in double), within the
//      derived float32 rounding tolerance. This is the sensitive check that catches kernel bugs;
//   2. every optimized backend vs the scalar kernel, within twice that tolerance;
//   3. quantized GEMV vs the float64 product of the original w and x, within the worst-case
//      quantization bound (a loose accuracy check, see test_helpers.hpp);
//   4. threaded vs single-threaded output, bit for bit.
#include <gtest/gtest.h>

#include <cmath>
#include <ostream>
#include <string>
#include <vector>

#include "int8k/int8k.hpp"
#include "test_helpers.hpp"

using namespace int8k;
using namespace int8k::testing;

namespace {

struct Case {
  std::size_t rows;
  std::size_t cols;
  float w_scale;
  float x_scale;
  std::uint64_t seed;
};

// Readable test output instead of gtest's raw byte dump of the parameter.
void PrintTo(const Case& c, std::ostream* os) {
  *os << c.rows << "x" << c.cols << " w_scale=" << c.w_scale << " x_scale=" << c.x_scale
      << " seed=" << c.seed;
}

std::vector<Case> make_cases(std::size_t count) {
  Rng rng(12345);
  std::vector<Case> cases;
  // Hand-picked boundary widths first.
  for (std::size_t cols : {1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 96u, 127u}) {
    cases.push_back({1 + cols % 5, cols, 1.0f, 1.0f, 1000 + cols});
  }
  // LLM layer widths (7B-class hidden and MLP sizes), where float accumulation error is largest.
  cases.push_back({4, 4096, 0.02f, 1.0f, 4096});
  cases.push_back({3, 11008, 0.02f, 1.0f, 11008});
  cases.push_back({2, 4096 + 17, 0.02f, 1.0f, 4113});
  while (cases.size() < count) {
    const float w_scale = std::pow(10.0f, rng.uniform(-4.0f, 2.0f));
    const float x_scale = std::pow(10.0f, rng.uniform(-3.0f, 3.0f));
    cases.push_back({rng.index(1, 80), rng.index(1, 700), w_scale, x_scale, rng.next_u64()});
  }
  return cases;
}

template <class QMatrix>
void check_case(const QMatrix& qw, const std::vector<float>& w, const std::vector<float>& x,
                const Case& c) {
  const auto exact = gemv_f64(w, c.rows, c.cols, x);
  const auto bound = gemv_error_bound(qw, w, x);
  const auto oracle = exact_oracle(qw, x);
  const auto tol = reorder_tolerance(qw, x);

  std::vector<float> scalar(c.rows);
  gemv(qw, x, scalar, {.backend = Backend::Scalar});
  for (std::size_t r = 0; r < c.rows; ++r) {
    ASSERT_NEAR(scalar[r], oracle.y[r], oracle.tol[r]) << "scalar vs oracle, row " << r;
    ASSERT_LE(std::fabs(scalar[r] - exact[r]), bound[r]) << "scalar vs f64, row " << r;
  }

  if (cpu_has_avx2()) {
    std::vector<float> avx(c.rows);
    gemv(qw, x, avx, {.backend = Backend::Avx2});
    for (std::size_t r = 0; r < c.rows; ++r) {
      ASSERT_NEAR(avx[r], oracle.y[r], oracle.tol[r]) << "avx2 vs oracle, row " << r;
      ASSERT_NEAR(avx[r], scalar[r], tol[r]) << "avx2 vs scalar, row " << r;
      ASSERT_LE(std::fabs(avx[r] - exact[r]), bound[r]) << "avx2 vs f64, row " << r;
    }
  }

  std::vector<float> threaded(c.rows);
  gemv(qw, x, threaded, {.backend = Backend::Auto, .threads = 4, .min_rows_per_thread = 1});
  std::vector<float> single(c.rows);
  gemv(qw, x, single, {.backend = Backend::Auto, .threads = 1});
  ASSERT_EQ(threaded, single);
}

class RandomShapes : public ::testing::TestWithParam<Case> {};

TEST_P(RandomShapes, Int8) {
  const Case c = GetParam();
  const auto w = random_normal(c.rows * c.cols, c.seed, c.w_scale);
  const auto x = random_normal(c.cols, c.seed + 1, c.x_scale);
  check_case(quantize_i8(MatrixView(w, c.rows, c.cols)), w, x, c);
}

TEST_P(RandomShapes, Int4) {
  const Case c = GetParam();
  const auto w = random_normal(c.rows * c.cols, c.seed, c.w_scale);
  const auto x = random_normal(c.cols, c.seed + 1, c.x_scale);
  check_case(quantize_q4(MatrixView(w, c.rows, c.cols)), w, x, c);
}

// Heavy-tailed inputs (a few large outliers per row) stress the per-block scale logic.
TEST_P(RandomShapes, Int4WithOutliers) {
  const Case c = GetParam();
  auto w = random_normal(c.rows * c.cols, c.seed, c.w_scale);
  Rng rng(c.seed ^ 0xABCDEF);
  for (std::size_t i = 0; i < w.size(); i += 1 + rng.index(0, 50)) w[i] *= 100.0f;
  const auto x = random_normal(c.cols, c.seed + 1, c.x_scale);
  check_case(quantize_q4(MatrixView(w, c.rows, c.cols)), w, x, c);
}

std::string case_name(const ::testing::TestParamInfo<Case>& info) {
  return "r" + std::to_string(info.param.rows) + "_c" + std::to_string(info.param.cols) + "_i" +
         std::to_string(info.index);
}

// 13 boundary widths + 3 LLM widths + 44 random shapes = 60 shapes, 180 test cases.
INSTANTIATE_TEST_SUITE_P(Property, RandomShapes, ::testing::ValuesIn(make_cases(60)), case_name);

}  // namespace
