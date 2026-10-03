// int8k-cli: quantize random matrices and inspect error / run a quick self-check.
//
//   int8k-cli info
//   int8k-cli quantize  [--rows N] [--cols N] [--format int8|q4] [--seed S] [--stddev X]
//   int8k-cli gemv      [--rows N] [--cols N] [--format int8|q4] [--backend B] [--threads T]
//   int8k-cli selfcheck [--rows N] [--cols N] [--threads T]
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include "int8k/int8k.hpp"

namespace {

using namespace int8k;

constexpr const char* kUsage =
    "usage: int8k-cli <command> [options]\n"
    "commands:\n"
    "  info       print CPU features and the backend 'auto' resolves to\n"
    "  quantize   quantize a random N(0, stddev^2) matrix and print reconstruction error\n"
    "  gemv       run one quantized GEMV and compare it with float32\n"
    "  selfcheck  compare every backend/thread count against the scalar reference\n"
    "options: --rows N --cols N --format int8|q4 --seed S --stddev X --backend auto|scalar|avx2\n"
    "         --threads T (0 = all hardware threads, at most 1024)\n";

// Parses the whole of `text` as an unsigned integer: no sign, no whitespace, no trailing junk.
std::optional<std::uint64_t> parse_u64(const std::string& text) {
  std::uint64_t value = 0;
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (text.empty() || ec != std::errc{} || ptr != end) return std::nullopt;
  return value;
}

// Parses "--key value" pairs. Unknown keys are an error so typos do not go unnoticed.
class Args {
 public:
  Args(int argc, char** argv, int first) {
    for (int i = first; i < argc; i += 2) {
      const std::string key = argv[i];
      if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
        throw std::invalid_argument("expected '--option value', got '" + key + "'");
      }
      values_[key.substr(2)] = argv[i + 1];
    }
  }

  /// A matrix dimension in [1, 2^20].
  std::size_t size(const std::string& key, std::size_t fallback) {
    return static_cast<std::size_t>(integer(key, fallback, 1, std::uint64_t{1} << 20));
  }
  /// Thread count in [0, 1024] (0 = one per hardware thread).
  unsigned threads(unsigned fallback) {
    return static_cast<unsigned>(integer("threads", fallback, 0, 1024));
  }
  /// Any 64-bit seed.
  std::uint64_t seed(std::uint64_t fallback) { return integer("seed", fallback, 0, UINT64_MAX); }
  /// A finite, strictly positive real number (the whole string must parse).
  float positive_real(const std::string& key, float fallback) {
    const auto v = take(key);
    if (!v) return fallback;
    char* end = nullptr;
    const float parsed = std::strtof(v->c_str(), &end);
    if (v->empty() || end != v->c_str() + v->size() || !std::isfinite(parsed) || parsed <= 0.0f) {
      throw std::invalid_argument("--" + key + " must be a finite number > 0, got '" + *v + "'");
    }
    return parsed;
  }
  std::string text(const std::string& key, const std::string& fallback) {
    const auto v = take(key);
    return v ? *v : fallback;
  }
  void finish() const {
    if (!values_.empty()) {
      throw std::invalid_argument("unknown option --" + values_.begin()->first);
    }
  }

 private:
  std::uint64_t integer(const std::string& key, std::uint64_t fallback, std::uint64_t lo,
                        std::uint64_t hi) {
    const auto v = take(key);
    if (!v) return fallback;
    const auto parsed = parse_u64(*v);
    if (!parsed || *parsed < lo || *parsed > hi) {
      throw std::invalid_argument("--" + key + " must be an integer in [" + std::to_string(lo) +
                                  ", " + std::to_string(hi) + "], got '" + *v + "'");
    }
    return *parsed;
  }

  std::optional<std::string> take(const std::string& key) {
    const auto it = values_.find(key);
    if (it == values_.end()) return std::nullopt;
    std::string value = it->second;
    values_.erase(it);
    return value;
  }

  std::map<std::string, std::string> values_;
};

enum class Format { Int8, Q4 };

Format parse_format(const std::string& s) {
  if (s == "int8") return Format::Int8;
  if (s == "q4") return Format::Q4;
  throw std::invalid_argument("--format must be int8 or q4, got '" + s + "'");
}

const char* format_name(Format f) {
  return f == Format::Int8 ? "int8" : "q4";
}

void print_stats(const ErrorStats& s) {
  std::printf("max_abs_error: %.6g\nrmse: %.6g\nsignal_rms: %.6g\nsnr_db: %.2f\n", s.max_abs_error,
              s.rmse, s.signal_rms, s.snr_db);
}

int cmd_info() {
  std::printf("int8k %s\n", "0.1.0");
  std::printf("avx2 available: %s\n", cpu_has_avx2() ? "yes" : "no");
  std::printf("auto backend: %s\n",
              std::string(backend_name(resolve_backend(Backend::Auto))).c_str());
  std::printf("hardware threads: %u\n", std::thread::hardware_concurrency());
  return 0;
}

int cmd_quantize(Args& args) {
  const std::size_t rows = args.size("rows", 256);
  const std::size_t cols = args.size("cols", 1024);
  const Format format = parse_format(args.text("format", "int8"));
  const std::uint64_t seed = args.seed(42);
  const float stddev = args.positive_real("stddev", 1.0f);
  args.finish();

  const auto w = random_normal(rows * cols, seed, stddev);
  const MatrixView view(w, rows, cols);
  std::vector<float> back;
  std::size_t bytes = 0;
  if (format == Format::Int8) {
    const auto q = quantize_i8(view);
    back = dequantize(q);
    bytes = q.storage_bytes();
  } else {
    const auto q = quantize_q4(view);
    back = dequantize(q);
    bytes = q.storage_bytes();
  }
  const std::size_t f32_bytes = w.size() * sizeof(float);
  std::printf("format: %s\nshape: %zux%zu\n", format_name(format), rows, cols);
  std::printf("f32_bytes: %zu\nquantized_bytes: %zu\ncompression: %.2fx\nbits_per_weight: %.3f\n",
              f32_bytes, bytes, double(f32_bytes) / double(bytes),
              8.0 * double(bytes) / double(w.size()));
  print_stats(compare(w, back));
  return 0;
}

int cmd_gemv(Args& args) {
  const std::size_t rows = args.size("rows", 4096);
  const std::size_t cols = args.size("cols", 4096);
  const Format format = parse_format(args.text("format", "int8"));
  const Backend backend = parse_backend(args.text("backend", "auto"));
  const unsigned threads = args.threads(1);
  args.finish();

  const auto w = random_normal(rows * cols, 1, 0.02f);
  const auto x = random_normal(cols, 2);
  std::vector<float> ref(rows), y(rows);
  gemv_f32(MatrixView(w, rows, cols), x, ref);

  // Weight quantization is an offline step (done once per model), so it is timed separately
  // and kept out of the GEMV timing. The GEMV timing includes activation quantization, which a
  // real decode step has to do on every call.
  using Ms = std::chrono::duration<double, std::milli>;
  const KernelOptions opts{.backend = backend, .threads = threads};
  const MatrixView wv(w, rows, cols);
  double weight_ms = 0.0;
  double gemv_ms = 0.0;
  auto run = [&](const auto& make_q) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto q = make_q(wv);
    const auto t1 = std::chrono::steady_clock::now();
    gemv(q, x, y, opts);
    const auto t2 = std::chrono::steady_clock::now();
    weight_ms = Ms(t1 - t0).count();
    gemv_ms = Ms(t2 - t1).count();
  };
  if (format == Format::Int8) {
    run([](MatrixView v) { return quantize_i8(v); });
  } else {
    run([](MatrixView v) { return quantize_q4(v); });
  }
  std::printf("format: %s\nbackend: %s\nthreads: %u\nshape: %zux%zu\n", format_name(format),
              std::string(backend_name(resolve_backend(backend))).c_str(), threads, rows, cols);
  std::printf("weight_quantize_ms (offline step): %.3f\n", weight_ms);
  std::printf("act_quantize+gemv_ms (single cold run, use int8k-bench for timing): %.3f\n",
              gemv_ms);
  print_stats(compare(ref, y));
  return 0;
}

// Runs every backend x format x thread count. Pass/fail is decided by exact checks only:
//   * every result must match the exact oracle sum(w^ x^) within the derived float32 rounding
//     tolerance (kernel_tolerance), which catches layout, scale and missing-block bugs;
//   * a threaded result must be bit-identical to the same backend single-threaded.
// The SNR against float32 is printed for information only: it measures quantization error,
// which is legitimately poor for tiny shapes (e.g. one row of 100 values), not kernel bugs.
template <class QMatrix>
bool selfcheck_format(const char* name, const QMatrix& q, std::span<const float> x,
                      std::span<const float> f32, std::span<const Backend> backends,
                      unsigned threads) {
  const auto ref = gemv_dequantized_f64(q, quantize_activations(x));
  const std::size_t blocks = blocks_for(q.cols);
  bool ok = true;
  for (const Backend b : backends) {
    std::vector<float> single(q.rows);
    gemv(q, x, single, {.backend = b, .threads = 1});
    for (const unsigned t : {1u, threads}) {
      std::vector<float> y(q.rows);
      gemv(q, x, y, {.backend = b, .threads = t, .min_rows_per_thread = 1});
      double worst = 0.0;  // largest |error| / tolerance over all rows
      double max_abs = 0.0;
      for (std::size_t r = 0; r < q.rows; ++r) {
        const double err = std::fabs(double(y[r]) - ref.y[r]);
        max_abs = std::max(max_abs, err);
        worst = std::max(worst, err / kernel_tolerance(blocks, ref.abs_sum[r]));
      }
      const bool pass = worst <= 1.0 && y == single;
      ok = ok && pass;
      std::printf(
          "%-5s %-6s threads=%-3u vs_oracle_max_abs=%.3g (%.2g of tolerance) "
          "snr_vs_f32_db=%.2f %s\n",
          name, std::string(backend_name(b)).c_str(), t, max_abs, worst, compare(f32, y).snr_db,
          pass ? "ok" : "MISMATCH");
    }
  }
  return ok;
}

int cmd_selfcheck(Args& args) {
  const std::size_t rows = args.size("rows", 257);
  const std::size_t cols = args.size("cols", 1000);
  const unsigned threads = args.threads(4);
  args.finish();

  const auto w = random_normal(rows * cols, 7, 0.05f);
  const auto x = random_normal(cols, 8);
  std::vector<float> f32(rows);
  gemv_f32(MatrixView(w, rows, cols), x, f32);

  std::vector<Backend> backends = {Backend::Scalar};
  if (cpu_has_avx2()) backends.push_back(Backend::Avx2);

  const bool ok_i8 =
      selfcheck_format("int8", quantize_i8(MatrixView(w, rows, cols)), x, f32, backends, threads);
  const bool ok_q4 =
      selfcheck_format("q4", quantize_q4(MatrixView(w, rows, cols)), x, f32, backends, threads);
  const bool ok = ok_i8 && ok_q4;
  std::printf("selfcheck: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fputs(kUsage, stderr);
    return 2;
  }
  const std::string_view command = argv[1];
  try {
    Args args(argc, argv, 2);
    if (command == "info") {
      args.finish();
      return cmd_info();
    }
    if (command == "quantize") return cmd_quantize(args);
    if (command == "gemv") return cmd_gemv(args);
    if (command == "selfcheck") return cmd_selfcheck(args);
    if (command == "help" || command == "--help" || command == "-h") {
      std::fputs(kUsage, stdout);
      return 0;
    }
    std::fprintf(stderr, "unknown command '%s'\n%s", argv[1], kUsage);
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
