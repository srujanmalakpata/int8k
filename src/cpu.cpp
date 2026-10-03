#include "int8k/cpu.hpp"

#include <stdexcept>
#include <string>

#include "kernels.hpp"

namespace int8k {

bool cpu_has_avx2() {
#if INT8K_X86
  // Cached: the answer cannot change while the process runs. libgcc's/compiler-rt's
  // implementation also checks that the OS saves the YMM registers (XGETBV), not just CPUID.
  static const bool has = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return has;
#else
  return false;
#endif
}

bool backend_available(Backend b) {
  switch (b) {
    case Backend::Auto:
    case Backend::Scalar:
      return true;
    case Backend::Avx2:
      return cpu_has_avx2();
  }
  return false;
}

Backend resolve_backend(Backend b) {
  if (b == Backend::Auto) {
    return cpu_has_avx2() ? Backend::Avx2 : Backend::Scalar;
  }
  if (!backend_available(b)) {
    throw std::invalid_argument("backend '" + std::string(backend_name(b)) +
                                "' is not available on this CPU/build");
  }
  return b;
}

std::string_view backend_name(Backend b) {
  switch (b) {
    case Backend::Auto:
      return "auto";
    case Backend::Scalar:
      return "scalar";
    case Backend::Avx2:
      return "avx2";
  }
  return "unknown";
}

Backend parse_backend(std::string_view name) {
  if (name == "auto") return Backend::Auto;
  if (name == "scalar") return Backend::Scalar;
  if (name == "avx2") return Backend::Avx2;
  throw std::invalid_argument("unknown backend '" + std::string(name) +
                              "' (expected auto, scalar or avx2)");
}

}  // namespace int8k
