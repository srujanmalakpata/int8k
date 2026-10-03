// Kernel-level tests: call the per-row dot-product kernels in src/kernels.hpp directly, and check
// which kernel the dispatcher picks for each Backend.
//
// The gemv/gemm tests go through select_i8/select_q4. If the dispatcher silently returned the
// scalar kernel for Backend::Avx2, every "AVX2 vs scalar" comparison there would compare scalar
// with itself and still pass. These tests close that gap: the dispatch test pins the function
// pointer, and the direct tests run the AVX2 kernels on hand-built blocks whose exact result is
// known, including the extreme codes (+/-127, nibble 0 = -8) and odd block counts.
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "int8k/cpu.hpp"
#include "int8k/quantize.hpp"
#include "int8k/random.hpp"
#include "kernels.hpp"

using namespace int8k;

namespace {

struct NamedI8 {
  const char* name;
  detail::DotI8Fn fn;
};
struct NamedQ4 {
  const char* name;
  detail::DotQ4Fn fn;
};

std::vector<NamedI8> i8_kernels() {
  std::vector<NamedI8> out = {{"scalar", detail::dot_i8_scalar}};
#if INT8K_X86
  if (cpu_has_avx2()) out.push_back({"avx2", detail::dot_i8_avx2});
#endif
  return out;
}

std::vector<NamedQ4> q4_kernels() {
  std::vector<NamedQ4> out = {{"scalar", detail::dot_q4_scalar}};
#if INT8K_X86
  if (cpu_has_avx2()) out.push_back({"avx2", detail::dot_q4_avx2});
#endif
  return out;
}

// Block scales that are powers of two, so every partial sum below is exactly representable in
// float (|sum| <= 7 * 516,128 in units of 0.25 < 2^24) and the kernels must match bit for bit.
float block_scale(std::size_t b) {
  constexpr float kScales[] = {1.0f, 0.5f, 0.25f, 2.0f};
  return kScales[b % 4];
}

// Expected sum_b ws * xs[b] * dot_b, computed with int64 block dots and double, independently
// of both kernels.
double expected_i8(const std::vector<std::int8_t>& w, float ws, const std::vector<std::int8_t>& x,
                   const std::vector<float>& xs) {
  double total = 0.0;
  for (std::size_t b = 0; b < xs.size(); ++b) {
    std::int64_t dot = 0;
    for (std::size_t j = 0; j < kBlockSize; ++j) {
      dot += std::int64_t{w[b * kBlockSize + j]} * std::int64_t{x[b * kBlockSize + j]};
    }
    total += double(ws) * double(xs[b]) * double(dot);
  }
  return total;
}

// Signed INT4 code of element j (0..31) of a 16-byte block: low nibbles hold elements 0..15,
// high nibbles hold 16..31, both biased by +8.
int q4_code(const std::uint8_t* block, std::size_t j) {
  const std::uint8_t byte = block[j % 16];
  return (j < 16 ? (byte & 0x0F) : (byte >> 4)) - 8;
}

double expected_q4(const std::vector<std::uint8_t>& wq, const std::vector<float>& ws,
                   const std::vector<std::int8_t>& x, const std::vector<float>& xs) {
  double total = 0.0;
  for (std::size_t b = 0; b < xs.size(); ++b) {
    std::int64_t dot = 0;
    for (std::size_t j = 0; j < kBlockSize; ++j) {
      dot += std::int64_t{q4_code(&wq[b * 16], j)} * std::int64_t{x[b * kBlockSize + j]};
    }
    total += double(ws[b]) * double(xs[b]) * double(dot);
  }
  return total;
}

// Activation code patterns. All use the extreme magnitude 127 except "random".
enum class Pattern { AllMax, AllMin, Alternating, Random, Zero };

std::vector<std::int8_t> make_codes(Pattern p, std::size_t n, std::uint64_t seed, int max_code) {
  std::vector<std::int8_t> v(n);
  Rng rng(seed);
  for (std::size_t i = 0; i < n; ++i) {
    int c = 0;
    switch (p) {
      case Pattern::AllMax:
        c = max_code;
        break;
      case Pattern::AllMin:
        c = -max_code;
        break;
      case Pattern::Alternating:
        c = (i % 2 == 0) ? max_code : -max_code;
        break;
      case Pattern::Random:
        c = static_cast<int>(rng.index(0, 2 * static_cast<std::size_t>(max_code))) - max_code;
        break;
      case Pattern::Zero:
        c = 0;
        break;
    }
    v[i] = static_cast<std::int8_t>(c);
  }
  return v;
}

constexpr Pattern kPatterns[] = {Pattern::AllMax, Pattern::AllMin, Pattern::Alternating,
                                 Pattern::Random, Pattern::Zero};
constexpr std::size_t kBlockCounts[] = {1, 2, 3, 4, 5, 7};

}  // namespace

TEST(Dispatch, ScalarBackendSelectsScalarKernels) {
  EXPECT_EQ(detail::select_i8(Backend::Scalar), &detail::dot_i8_scalar);
  EXPECT_EQ(detail::select_q4(Backend::Scalar), &detail::dot_q4_scalar);
}

TEST(Dispatch, Avx2BackendSelectsAvx2Kernels) {
#if INT8K_X86
  if (!cpu_has_avx2()) GTEST_SKIP() << "AVX2 not available on this CPU";
  EXPECT_EQ(detail::select_i8(Backend::Avx2), &detail::dot_i8_avx2);
  EXPECT_EQ(detail::select_q4(Backend::Avx2), &detail::dot_q4_avx2);
#else
  EXPECT_THROW((void)detail::select_i8(Backend::Avx2), std::invalid_argument);
  EXPECT_THROW((void)detail::select_q4(Backend::Avx2), std::invalid_argument);
#endif
}

TEST(Dispatch, AutoSelectsTheFastestAvailableKernel) {
#if INT8K_X86
  if (cpu_has_avx2()) {
    EXPECT_EQ(detail::select_i8(Backend::Auto), &detail::dot_i8_avx2);
    EXPECT_EQ(detail::select_q4(Backend::Auto), &detail::dot_q4_avx2);
    return;
  }
#endif
  EXPECT_EQ(detail::select_i8(Backend::Auto), &detail::dot_i8_scalar);
  EXPECT_EQ(detail::select_q4(Backend::Auto), &detail::dot_q4_scalar);
}

TEST(Kernels, Int8DotIsExactOnExtremeAndRandomCodes) {
  for (const auto& k : i8_kernels()) {
    for (std::size_t blocks : kBlockCounts) {
      const std::size_t n = blocks * kBlockSize;
      std::vector<float> xs(blocks);
      for (std::size_t b = 0; b < blocks; ++b) xs[b] = block_scale(b);
      for (Pattern wp : kPatterns) {
        for (Pattern xp : kPatterns) {
          const auto w = make_codes(wp, n, 100 + blocks, kI8Max);
          const auto x = make_codes(xp, n, 200 + blocks, kI8Max);
          const float got = k.fn(w.data(), 0.5f, x.data(), xs.data(), blocks);
          EXPECT_EQ(double(got), expected_i8(w, 0.5f, x, xs))
              << k.name << " blocks=" << blocks << " w=" << int(wp) << " x=" << int(xp);
        }
      }
    }
  }
}

TEST(Kernels, Q4DotIsExactOnExtremeAndRandomNibbles) {
  // Every nibble value 0..15 (codes -8..7), plus all-0x00 (every code -8, the one asymmetric
  // value) and all-0xFF (every code +7).
  for (const auto& k : q4_kernels()) {
    for (std::size_t blocks : kBlockCounts) {
      std::vector<float> ws(blocks), xs(blocks);
      for (std::size_t b = 0; b < blocks; ++b) {
        ws[b] = block_scale(b + 1);
        xs[b] = block_scale(b);
      }
      Rng rng(blocks);
      std::vector<std::vector<std::uint8_t>> weights = {
          std::vector<std::uint8_t>(blocks * 16, 0x00),
          std::vector<std::uint8_t>(blocks * 16, 0xFF),
          std::vector<std::uint8_t>(blocks * 16, 0x0F), std::vector<std::uint8_t>(blocks * 16)};
      for (auto& byte : weights[3]) byte = static_cast<std::uint8_t>(rng.index(0, 255));
      for (std::size_t wi = 0; wi < weights.size(); ++wi) {
        for (Pattern xp : kPatterns) {
          const auto x = make_codes(xp, blocks * kBlockSize, 300 + blocks, kI8Max);
          const float got = k.fn(weights[wi].data(), ws.data(), x.data(), xs.data(), blocks);
          EXPECT_EQ(double(got), expected_q4(weights[wi], ws, x, xs))
              << k.name << " blocks=" << blocks << " weights#" << wi << " x=" << int(xp);
        }
      }
    }
  }
}

TEST(Kernels, Q4NibbleLayoutLowHalfThenHighHalf) {
  // One weight element at a time set to +7 (all others 0 = nibble 8), activation j = j + 1, so the
  // result names exactly which activation element the kernel paired with weight element e.
  std::vector<std::int8_t> x(kBlockSize);
  for (std::size_t j = 0; j < kBlockSize; ++j) x[j] = static_cast<std::int8_t>(j + 1);
  const float one = 1.0f;
  for (const auto& k : q4_kernels()) {
    for (std::size_t e = 0; e < kBlockSize; ++e) {
      std::vector<std::uint8_t> wq(16, 0x88);
      if (e < 16) {
        wq[e] = static_cast<std::uint8_t>((wq[e] & 0xF0) | 15);
      } else {
        wq[e - 16] = static_cast<std::uint8_t>((wq[e - 16] & 0x0F) | (15 << 4));
      }
      EXPECT_EQ(k.fn(wq.data(), &one, x.data(), &one, 1), 7.0f * float(e + 1))
          << k.name << " element " << e;
    }
  }
}

TEST(Kernels, ZeroBlocksGiveZero) {
  const std::int8_t dummy_i8 = 0;
  const std::uint8_t dummy_u8 = 0;
  const float s = 1.0f;
  for (const auto& k : i8_kernels()) EXPECT_EQ(k.fn(&dummy_i8, s, &dummy_i8, &s, 0), 0.0f);
  for (const auto& k : q4_kernels()) EXPECT_EQ(k.fn(&dummy_u8, &s, &dummy_i8, &s, 0), 0.0f);
}
