// Deterministic pseudo-random data for tests, benchmarks and the CLI.
//
// std::normal_distribution is implementation-defined (libstdc++ and libc++ produce different
// streams), so we use a tiny self-contained generator to get identical data on every platform.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace int8k {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : state_(seed ^ 0x9E3779B97F4A7C15ULL) {}

  /// SplitMix64 step.
  std::uint64_t next_u64();
  /// Uniform in [0, 1).
  double uniform();
  /// Uniform in [lo, hi) for lo < hi (24 random bits; lo + u * (hi - lo) can still round up to
  /// hi when |lo| is much larger than hi - lo).
  float uniform(float lo, float hi);
  /// Standard normal via Box-Muller.
  float normal(float mean = 0.0f, float stddev = 1.0f);
  /// Uniform integer in [lo, hi] (requires lo <= hi; modulo bias is negligible for tests).
  std::size_t index(std::size_t lo, std::size_t hi);

 private:
  std::uint64_t state_;
};

/// `n` samples from N(0, stddev^2).
[[nodiscard]] std::vector<float> random_normal(std::size_t n, std::uint64_t seed,
                                               float stddev = 1.0f);

}  // namespace int8k
