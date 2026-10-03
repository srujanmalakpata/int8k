// Error statistics used by the CLI, benchmarks and tests.
#pragma once

#include <cstddef>
#include <span>

namespace int8k {

struct ErrorStats {
  double max_abs_error = 0.0;
  double rmse = 0.0;
  double signal_rms = 0.0;
  /// 10 * log10(signal power / noise power); +inf when the error is exactly zero.
  double snr_db = 0.0;
  std::size_t count = 0;
};

/// Compares `approx` against `reference` element-wise. Sizes must match.
[[nodiscard]] ErrorStats compare(std::span<const float> reference, std::span<const float> approx);

}  // namespace int8k
