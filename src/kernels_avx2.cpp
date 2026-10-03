// AVX2 kernels.
//
// We use __attribute__((target("avx2,fma"))) on each function instead of compiling this file
// with -mavx2. With -mavx2, inline functions from headers (std::vector, std::span...) emitted in
// this translation unit would also be AVX2 code, and the linker may pick that copy for the whole
// program, crashing on CPUs without AVX2. The target attribute confines AVX2 to these functions.
#include "kernels.hpp"

#if INT8K_X86

#include <immintrin.h>

#include "int8k/quantize.hpp"

#define INT8K_TARGET_AVX2 __attribute__((target("avx2,fma")))

namespace int8k::detail {
namespace {

// Dot product of 32 signed int8 pairs, returned as 8 int32 partial sums.
//
// AVX2 has no signed x signed 8-bit multiply-add. _mm256_maddubs_epi16 multiplies UNSIGNED bytes
// by SIGNED bytes, so we move the sign of x onto w: x * w == |x| * (w * sign(x)).
// Pairwise int16 sums are at most 2 * 127 * 127 = 32,258 < 32,767, so maddubs never saturates
// (this is why codes are restricted to [-127, 127] and never use -128).
INT8K_TARGET_AVX2 inline __m256i dot32_epi32(__m256i w, __m256i x) {
  const __m256i x_abs = _mm256_sign_epi8(x, x);
  const __m256i w_signed = _mm256_sign_epi8(w, x);
  const __m256i pairs16 = _mm256_maddubs_epi16(x_abs, w_signed);
  return _mm256_madd_epi16(pairs16, _mm256_set1_epi16(1));
}

INT8K_TARGET_AVX2 inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// Unpacks 16 bytes of Q4 data into 32 signed codes in [-8, 7]: low nibbles are elements 0..15,
// high nibbles are elements 16..31 (matches the layout written by quantize_q4).
INT8K_TARGET_AVX2 inline __m256i unpack_q4(const std::uint8_t* bytes) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bytes));
  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(raw, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask);
  const __m256i codes = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
  return _mm256_sub_epi8(codes, _mm256_set1_epi8(8));
}

INT8K_TARGET_AVX2 inline __m256i load32(const std::int8_t* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

}  // namespace

INT8K_TARGET_AVX2 float dot_f32_avx2(const float* w, const float* x, std::size_t n) {
  // Four independent 8-wide accumulators (32 floats per iteration) to cover FMA latency.
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24), acc3);
  }
  for (; i + 8 <= n; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc0);
  }
  float sum = hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
  for (; i < n; ++i) sum += w[i] * x[i];  // unpadded float rows: scalar tail
  return sum;
}

INT8K_TARGET_AVX2 float dot_i8_avx2(const std::int8_t* w, float ws, const std::int8_t* x,
                                    const float* xs, std::size_t blocks) {
  // Two accumulators split the FMA dependency chain in two. The loop is limited by the integer
  // sign/maddubs/madd/convert work per block and by memory bandwidth, not by FMA latency.
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t b = 0;
  for (; b + 2 <= blocks; b += 2) {
    const __m256i d0 = dot32_epi32(load32(w + b * kBlockSize), load32(x + b * kBlockSize));
    const __m256i d1 =
        dot32_epi32(load32(w + (b + 1) * kBlockSize), load32(x + (b + 1) * kBlockSize));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ws * xs[b]), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(ws * xs[b + 1]), acc1);
  }
  if (b < blocks) {
    const __m256i d0 = dot32_epi32(load32(w + b * kBlockSize), load32(x + b * kBlockSize));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ws * xs[b]), acc0);
  }
  return hsum(_mm256_add_ps(acc0, acc1));
}

INT8K_TARGET_AVX2 float dot_q4_avx2(const std::uint8_t* wq, const float* ws, const std::int8_t* x,
                                    const float* xs, std::size_t blocks) {
  constexpr std::size_t kStep = QuantizedQ4Matrix::kBytesPerBlock;
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t b = 0;
  for (; b + 2 <= blocks; b += 2) {
    const __m256i d0 = dot32_epi32(unpack_q4(wq + b * kStep), load32(x + b * kBlockSize));
    const __m256i d1 =
        dot32_epi32(unpack_q4(wq + (b + 1) * kStep), load32(x + (b + 1) * kBlockSize));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ws[b] * xs[b]), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(ws[b + 1] * xs[b + 1]), acc1);
  }
  if (b < blocks) {
    const __m256i d0 = dot32_epi32(unpack_q4(wq + b * kStep), load32(x + b * kBlockSize));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ws[b] * xs[b]), acc0);
  }
  return hsum(_mm256_add_ps(acc0, acc1));
}

}  // namespace int8k::detail

#endif  // INT8K_X86
