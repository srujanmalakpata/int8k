// Runtime CPU feature detection and kernel backend selection.
#pragma once

#include <string_view>

namespace int8k {

/// Which kernel implementation to run.
enum class Backend {
  Auto,    ///< Fastest backend available on this CPU (resolved at call time).
  Scalar,  ///< Portable C++ reference; always available, used as the correctness oracle.
  Avx2,    ///< x86-64 AVX2 + FMA intrinsics; only available when the CPU reports both.
};

/// True when this binary was built for x86 and the running CPU supports AVX2 and FMA.
/// Always false on other architectures (e.g. ARM / Apple silicon).
[[nodiscard]] bool cpu_has_avx2();

/// True if `b` can run on this machine (Auto and Scalar are always available).
[[nodiscard]] bool backend_available(Backend b);

/// Maps Auto to a concrete backend. Throws std::invalid_argument for an unavailable backend.
[[nodiscard]] Backend resolve_backend(Backend b);

[[nodiscard]] std::string_view backend_name(Backend b);

/// Parses "auto", "scalar" or "avx2". Throws std::invalid_argument otherwise.
[[nodiscard]] Backend parse_backend(std::string_view name);

}  // namespace int8k
