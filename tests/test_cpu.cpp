#include <gtest/gtest.h>

#include <stdexcept>

#include "int8k/cpu.hpp"

using namespace int8k;

TEST(Cpu, ScalarAndAutoAreAlwaysAvailable) {
  EXPECT_TRUE(backend_available(Backend::Scalar));
  EXPECT_TRUE(backend_available(Backend::Auto));
  EXPECT_EQ(resolve_backend(Backend::Scalar), Backend::Scalar);
}

TEST(Cpu, AutoResolvesToFastestAvailable) {
  const Backend expected = cpu_has_avx2() ? Backend::Avx2 : Backend::Scalar;
  EXPECT_EQ(resolve_backend(Backend::Auto), expected);
}

TEST(Cpu, RequestingUnavailableAvx2Throws) {
  if (cpu_has_avx2()) {
    EXPECT_EQ(resolve_backend(Backend::Avx2), Backend::Avx2);
  } else {
    EXPECT_THROW((void)resolve_backend(Backend::Avx2), std::invalid_argument);
  }
}

TEST(Cpu, ParseAndNameRoundTrip) {
  for (Backend b : {Backend::Auto, Backend::Scalar, Backend::Avx2}) {
    EXPECT_EQ(parse_backend(backend_name(b)), b);
  }
  EXPECT_THROW((void)parse_backend("neon"), std::invalid_argument);
  EXPECT_THROW((void)parse_backend(""), std::invalid_argument);
}
