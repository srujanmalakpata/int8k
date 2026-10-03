#include "int8k/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace int8k {

ErrorStats compare(std::span<const float> reference, std::span<const float> approx) {
  if (reference.size() != approx.size()) {
    throw std::invalid_argument("compare: size mismatch");
  }
  ErrorStats s;
  s.count = reference.size();
  if (s.count == 0) return s;

  double noise = 0.0;
  double signal = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double ref = reference[i];
    const double err = static_cast<double>(approx[i]) - ref;
    s.max_abs_error = std::max(s.max_abs_error, std::fabs(err));
    noise += err * err;
    signal += ref * ref;
  }
  const auto n = static_cast<double>(s.count);
  s.rmse = std::sqrt(noise / n);
  s.signal_rms = std::sqrt(signal / n);
  s.snr_db =
      noise == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(signal / noise);
  return s;
}

}  // namespace int8k
