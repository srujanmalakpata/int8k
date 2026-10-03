// int8k-bench: times float32 vs quantized GEMV/GEMM on LLM-like layer shapes.
//
//   int8k-bench [--threads T] [--reps R] [--quick]     (T = 0: one per hardware thread)
//
// Shapes follow a 7B-class decoder layer (hidden 4096, MLP 11008), written as rows x cols =
// out_features x in_features. Every quantized timing INCLUDES on-the-fly activation
// quantization, as a real decode step would. Reported time is the median of R repetitions after
// warm-up; GFLOP/s counts 2 * rows * cols (* batch) flops; GB/s counts weight bytes read.
// "vs f32" is relative to the single-threaded float32 GEMV/GEMM using the best available
// instruction set (AVX2 FMA on x86), not to the naive scalar loop.
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "int8k/int8k.hpp"

namespace {

using namespace int8k;
using Clock = std::chrono::steady_clock;

struct Config {
  unsigned threads = 4;
  int reps = 15;
  bool quick = false;
};

struct Shape {
  const char* name;
  std::size_t rows;
  std::size_t cols;
};

double median_ms(int reps, const std::function<void()>& fn) {
  fn();  // warm-up: page in buffers, spin up caches
  fn();
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(reps));
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    fn();
    times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

void print_row(const char* shape, const char* kernel, const char* backend, unsigned threads,
               std::size_t batch, double ms, double flops, std::size_t weight_bytes,
               double baseline_ms) {
  const double gflops = flops / (ms * 1e6);
  const double gbps = double(weight_bytes) / (ms * 1e6);
  std::printf("| %-12s | %-10s | %-6s | %7u | %5zu | %9.3f | %8.2f | %7.2f | %7.2fx |\n", shape,
              kernel, backend, threads, batch, ms, gflops, gbps, baseline_ms / ms);
}

void bench_gemv(const Shape& s, const Config& cfg) {
  const auto w = random_normal(s.rows * s.cols, 1, 0.02f);
  const auto x = random_normal(s.cols, 2);
  std::vector<float> y(s.rows);
  const MatrixView wv(w, s.rows, s.cols);
  const double flops = 2.0 * double(s.rows) * double(s.cols);

  // Baseline for the "vs f32" column: the best single-threaded float32 GEMV on this CPU
  // (AVX2 FMA when available), so speedups compare like with like.
  const Backend best = resolve_backend(Backend::Auto);
  const std::string best_name(backend_name(best));
  const double f32_naive_ms = median_ms(cfg.reps, [&] { gemv_f32(wv, x, y, Backend::Scalar); });
  const double f32_ms = best == Backend::Scalar
                            ? f32_naive_ms
                            : median_ms(cfg.reps, [&] { gemv_f32(wv, x, y, best); });
  print_row(s.name, "f32 gemv", "scalar", 1, 1, f32_naive_ms, flops, w.size() * sizeof(float),
            f32_ms);
  if (best != Backend::Scalar) {
    print_row(s.name, "f32 gemv", best_name.c_str(), 1, 1, f32_ms, flops, w.size() * sizeof(float),
              f32_ms);
  }

  const auto qi = quantize_i8(wv);
  const auto q4 = quantize_q4(wv);
  std::vector<Backend> backends = {Backend::Scalar};
  if (cpu_has_avx2()) backends.push_back(Backend::Avx2);

  for (const Backend b : backends) {
    const std::string name(backend_name(b));
    for (const unsigned t : {1u, cfg.threads}) {
      const KernelOptions opts{.backend = b, .threads = t};
      const double ms_i8 = median_ms(cfg.reps, [&] { gemv(qi, x, y, opts); });
      print_row(s.name, "int8 gemv", name.c_str(), t, 1, ms_i8, flops, qi.storage_bytes(), f32_ms);
      const double ms_q4 = median_ms(cfg.reps, [&] { gemv(q4, x, y, opts); });
      print_row(s.name, "q4 gemv", name.c_str(), t, 1, ms_q4, flops, q4.storage_bytes(), f32_ms);
      if (cfg.threads == 1) break;
    }
  }
}

void bench_gemm(const Shape& s, std::size_t batch, const Config& cfg) {
  const auto w = random_normal(s.rows * s.cols, 3, 0.02f);
  const auto x = random_normal(batch * s.cols, 4);
  std::vector<float> y(batch * s.rows);
  const MatrixView wv(w, s.rows, s.cols);
  const MatrixView xv(x, batch, s.cols);
  const double flops = 2.0 * double(s.rows) * double(s.cols) * double(batch);

  const Backend best = resolve_backend(Backend::Auto);
  const std::string name(backend_name(best));
  const double f32_ms = median_ms(cfg.reps, [&] { gemm_f32(wv, xv, y, best); });
  print_row(s.name, "f32 gemm", name.c_str(), 1, batch, f32_ms, flops, w.size() * sizeof(float),
            f32_ms);
  const auto qi = quantize_i8(wv);
  const auto q4 = quantize_q4(wv);
  for (const unsigned t : {1u, cfg.threads}) {
    const KernelOptions opts{.backend = best, .threads = t};
    const double ms_i8 = median_ms(cfg.reps, [&] { gemm(qi, xv, y, opts); });
    print_row(s.name, "int8 gemm", name.c_str(), t, batch, ms_i8, flops, qi.storage_bytes(),
              f32_ms);
    const double ms_q4 = median_ms(cfg.reps, [&] { gemm(q4, xv, y, opts); });
    print_row(s.name, "q4 gemm", name.c_str(), t, batch, ms_q4, flops, q4.storage_bytes(), f32_ms);
    if (cfg.threads == 1) break;
  }
}

[[noreturn]] void usage_error(const std::string& message) {
  std::fprintf(stderr, "error: %s\nusage: int8k-bench [--threads T] [--reps R] [--quick]\n",
               message.c_str());
  std::exit(2);
}

// Parses the whole of `text` as an integer in [lo, hi] (no sign, no trailing junk), the same
// strict rule int8k-cli uses.
std::uint64_t parse_int(std::string_view flag, std::string_view text, std::uint64_t lo,
                        std::uint64_t hi) {
  std::uint64_t value = 0;
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (text.empty() || ec != std::errc{} || ptr != end || value < lo || value > hi) {
    usage_error(std::string(flag) + " must be an integer in [" + std::to_string(lo) + ", " +
                std::to_string(hi) + "], got '" + std::string(text) + "'");
  }
  return value;
}

Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--quick") {
      cfg.quick = true;
    } else if (a == "--threads" && i + 1 < argc) {
      // 0 = one thread per hardware thread, as in KernelOptions.
      const auto t = static_cast<unsigned>(parse_int(a, argv[++i], 0, 1024));
      cfg.threads = t == 0 ? std::max(1u, std::thread::hardware_concurrency()) : t;
    } else if (a == "--reps" && i + 1 < argc) {
      cfg.reps = static_cast<int>(parse_int(a, argv[++i], 1, 100000));
    } else {
      usage_error("unexpected argument '" + std::string(a) + "'");
    }
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  const Config cfg = parse(argc, argv);
  std::printf("int8k-bench: avx2=%s hardware_threads=%u threads=%u reps=%d (median ms)\n\n",
              cpu_has_avx2() ? "yes" : "no", std::thread::hardware_concurrency(), cfg.threads,
              cfg.reps);
  std::printf("| %-12s | %-10s | %-6s | %7s | %5s | %9s | %8s | %7s | %8s |\n", "shape", "kernel",
              "impl", "threads", "batch", "ms", "GFLOP/s", "W GB/s", "vs f32");
  std::printf(
      "|--------------|------------|--------|---------|-------|-----------|----------|"
      "---------|----------|\n");

  const std::vector<Shape> shapes = cfg.quick ? std::vector<Shape>{{"512x1024", 512, 1024}}
                                              : std::vector<Shape>{{"4096x4096", 4096, 4096},
                                                                   {"11008x4096", 11008, 4096},
                                                                   {"4096x11008", 4096, 11008}};
  for (const auto& s : shapes) bench_gemv(s, cfg);
  bench_gemm(cfg.quick ? Shape{"512x1024", 512, 1024} : Shape{"4096x4096", 4096, 4096}, 8, cfg);
  return 0;
}
