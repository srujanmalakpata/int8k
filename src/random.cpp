#include "int8k/random.hpp"

#include <cmath>
#include <numbers>

namespace int8k {

std::uint64_t Rng::next_u64() {
  std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double Rng::uniform() {
  // Top 53 bits -> double in [0, 1).
  return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
}

float Rng::uniform(float lo, float hi) {
  // Top 24 bits -> float in [0, 1) exactly. Rounding a [0, 1) double to float could give 1.0f.
  const float u = static_cast<float>(next_u64() >> 40) * 0x1.0p-24f;
  return lo + u * (hi - lo);
}

float Rng::normal(float mean, float stddev) {
  const double u1 = 1.0 - uniform();  // (0, 1], so log() is finite
  const double u2 = uniform();
  const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
  return mean + stddev * static_cast<float>(z);
}

std::size_t Rng::index(std::size_t lo, std::size_t hi) {
  const std::uint64_t span = static_cast<std::uint64_t>(hi - lo) + 1;  // 0 means the full range
  return lo + static_cast<std::size_t>(span == 0 ? next_u64() : next_u64() % span);
}

std::vector<float> random_normal(std::size_t n, std::uint64_t seed, float stddev) {
  Rng rng(seed);
  std::vector<float> out(n);
  for (auto& v : out) v = rng.normal(0.0f, stddev);
  return out;
}

}  // namespace int8k
