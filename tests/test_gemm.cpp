#include <gtest/gtest.h>

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "int8k/int8k.hpp"
#include "test_helpers.hpp"

using namespace int8k;
using namespace int8k::testing;

namespace {

// GEMM must equal m independent GEMVs exactly: same kernel, same per-row arithmetic.
template <class QMatrix>
void expect_gemm_equals_stacked_gemv(const QMatrix& qw, const std::vector<float>& x, std::size_t m,
                                     const KernelOptions& opts) {
  const std::size_t n = qw.rows, k = qw.cols;
  std::vector<float> y(m * n, -1.0f);
  gemm(qw, MatrixView(x, m, k), y, opts);
  for (std::size_t i = 0; i < m; ++i) {
    std::vector<float> row(n);
    gemv(qw, std::span<const float>(x).subspan(i * k, k), row, opts);
    EXPECT_EQ(std::vector<float>(y.begin() + std::ptrdiff_t(i * n),
                                 y.begin() + std::ptrdiff_t((i + 1) * n)),
              row)
        << "batch row " << i;
  }
}

}  // namespace

TEST(Gemm, EqualsStackedGemvForAllBackendsAndThreads) {
  const std::size_t n = 70, k = 200, m = 5;
  const auto w = random_normal(n * k, 60);
  const auto x = random_normal(m * k, 61);
  const auto qi = quantize_i8(MatrixView(w, n, k));
  const auto q4 = quantize_q4(MatrixView(w, n, k));
  std::vector<Backend> backends = {Backend::Scalar};
  if (cpu_has_avx2()) backends.push_back(Backend::Avx2);
  for (Backend b : backends) {
    for (unsigned t : {0u, 1u, 3u}) {
      const KernelOptions opts{.backend = b, .threads = t, .min_rows_per_thread = 1};
      expect_gemm_equals_stacked_gemv(qi, x, m, opts);
      expect_gemm_equals_stacked_gemv(q4, x, m, opts);
    }
  }
}

// Each batch row of the GEMM must match the exact oracle for that activation row; this checks
// the GEMM driver's own indexing independently of gemv.
TEST(Gemm, MatchesExactOraclePerBatchRow) {
  const std::size_t n = 33, k = 4096 + 40, m = 3;
  const auto w = random_normal(n * k, 80, 0.02f);
  const auto x = random_normal(m * k, 81);
  const auto qi = quantize_i8(MatrixView(w, n, k));
  const auto q4 = quantize_q4(MatrixView(w, n, k));
  std::vector<Backend> backends = {Backend::Scalar};
  if (cpu_has_avx2()) backends.push_back(Backend::Avx2);
  for (Backend b : backends) {
    std::vector<float> yi(m * n), y4(m * n);
    gemm(qi, MatrixView(x, m, k), yi, {.backend = b, .threads = 2, .min_rows_per_thread = 1});
    gemm(q4, MatrixView(x, m, k), y4, {.backend = b, .threads = 2, .min_rows_per_thread = 1});
    for (std::size_t i = 0; i < m; ++i) {
      const auto xi = std::span<const float>(x).subspan(i * k, k);
      const auto oi = exact_oracle(qi, xi);
      const auto o4 = exact_oracle(q4, xi);
      for (std::size_t r = 0; r < n; ++r) {
        EXPECT_NEAR(yi[i * n + r], oi.y[r], oi.tol[r]) << backend_name(b) << " int8 " << i;
        EXPECT_NEAR(y4[i * n + r], o4.y[r], o4.tol[r]) << backend_name(b) << " q4 " << i;
      }
    }
  }
}

TEST(Gemm, CloseToFloat32Gemm) {
  const std::size_t n = 64, k = 512, m = 4;
  const auto w = random_normal(n * k, 70, 0.05f);
  const auto x = random_normal(m * k, 71);
  std::vector<float> ref(m * n), yi(m * n), y4(m * n);
  gemm_f32(MatrixView(w, n, k), MatrixView(x, m, k), ref);
  gemm(quantize_i8(MatrixView(w, n, k)), MatrixView(x, m, k), yi);
  gemm(quantize_q4(MatrixView(w, n, k)), MatrixView(x, m, k), y4);
  EXPECT_GT(compare(ref, yi).snr_db, 30.0);
  EXPECT_GT(compare(ref, y4).snr_db, 12.0);
}

TEST(Gemm, EmptyBatchIsANoOp) {
  const std::vector<float> w(4 * 10, 1.0f);
  const std::vector<float> none;
  std::vector<float> y;
  EXPECT_NO_THROW(gemm(quantize_i8(MatrixView(w, 4, 10)), MatrixView(none, 0, 10), y));
  EXPECT_NO_THROW(gemm(quantize_q4(MatrixView(w, 4, 10)), MatrixView(none, 0, 10), y));
  std::vector<float> y_bad(1);
  EXPECT_THROW(gemm(quantize_i8(MatrixView(w, 4, 10)), MatrixView(none, 0, 10), y_bad),
               std::invalid_argument);
}

TEST(Gemm, ShapeMismatchesThrow) {
  const std::vector<float> w(4 * 10, 1.0f), x(2 * 10, 1.0f), x_bad(2 * 9, 1.0f);
  const auto qi = quantize_i8(MatrixView(w, 4, 10));
  std::vector<float> y(8), y_bad(7);
  EXPECT_THROW(gemm(qi, MatrixView(x_bad, 2, 9), y), std::invalid_argument);
  EXPECT_THROW(gemm(qi, MatrixView(x, 2, 10), y_bad), std::invalid_argument);
}

TEST(ReferenceF32, GemvComputesPlainDotProducts) {
  const std::vector<float> w = {1, 2, 3, 4, 5, 6};
  const std::vector<float> x = {1, 0, -1};
  std::vector<float> y(2);
  gemv_f32(MatrixView(w, 2, 3), x, y);
  EXPECT_EQ(y, (std::vector<float>{-2, -2}));
}

TEST(ReferenceF32, Avx2MatchesScalarForAllTailLengths) {
  if (!cpu_has_avx2()) GTEST_SKIP() << "AVX2 not available on this CPU";
  for (std::size_t cols : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 100u, 1029u}) {
    const std::size_t rows = 5;
    const auto w = random_normal(rows * cols, cols);
    const auto x = random_normal(cols, cols + 1);
    std::vector<float> s(rows), v(rows);
    gemv_f32(MatrixView(w, rows, cols), x, s, Backend::Scalar);
    gemv_f32(MatrixView(w, rows, cols), x, v, Backend::Avx2);
    for (std::size_t r = 0; r < rows; ++r) {
      double l1 = 0.0;
      for (std::size_t c = 0; c < cols; ++c) l1 += std::fabs(double(w[r * cols + c]) * x[c]);
      EXPECT_NEAR(s[r], v[r], 1e-5 * l1) << "cols=" << cols << " row=" << r;
    }
  }
}
