#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "int8k/random.hpp"

using namespace int8k;

TEST(Rng, SameSeedGivesSameStream) {
  EXPECT_EQ(random_normal(100, 7), random_normal(100, 7));
  EXPECT_NE(random_normal(100, 7), random_normal(100, 8));
}

TEST(Rng, UniformStaysInHalfOpenRange) {
  Rng rng(1);
  for (int i = 0; i < 100000; ++i) {
    const float u = rng.uniform(0.0f, 1.0f);
    ASSERT_GE(u, 0.0f);
    ASSERT_LT(u, 1.0f);
  }
}

TEST(Rng, IndexCoversInclusiveRangeAndFullRange) {
  Rng rng(2);
  bool saw_lo = false, saw_hi = false;
  for (int i = 0; i < 1000; ++i) {
    const std::size_t v = rng.index(3, 5);
    ASSERT_GE(v, 3u);
    ASSERT_LE(v, 5u);
    saw_lo = saw_lo || v == 3;
    saw_hi = saw_hi || v == 5;
  }
  EXPECT_TRUE(saw_lo && saw_hi);
  // hi - lo + 1 wraps to 0 for the full range; handle it without computing x % 0.
  (void)rng.index(0, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(rng.index(9, 9), 9u);
}
