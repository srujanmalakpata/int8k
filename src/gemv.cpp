#include "int8k/gemv.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "parallel.hpp"

namespace int8k {

namespace detail {

DotI8Fn select_i8(Backend b) {
  switch (resolve_backend(b)) {
#if INT8K_X86
    case Backend::Avx2:
      return dot_i8_avx2;
#endif
    default:
      return dot_i8_scalar;
  }
}

DotQ4Fn select_q4(Backend b) {
  switch (resolve_backend(b)) {
#if INT8K_X86
    case Backend::Avx2:
      return dot_q4_avx2;
#endif
    default:
      return dot_q4_scalar;
  }
}

}  // namespace detail

namespace {

using detail::DotI8Fn;
using detail::DotQ4Fn;

void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}

void check_activation(const QuantizedActivations& x, std::size_t cols) {
  require(x.length == cols, "gemv: activation length must equal weight cols");
  require(x.values.size() == blocks_for(cols) * kBlockSize && x.scales.size() == blocks_for(cols),
          "gemv: malformed QuantizedActivations");
}

// The quantized formats are public aggregates, so a caller can build one by hand. The kernels
// read raw pointers, so every buffer size is checked here (O(1)) before any kernel runs. Code
// values themselves are not scanned (that would be a full extra pass over the weights); see the
// precondition in quantize.hpp.
void check_weights(const QuantizedI8Matrix& w) {
  require(w.row_stride == blocks_for(w.cols) * kBlockSize,
          "gemv: malformed QuantizedI8Matrix (row_stride)");
  require(w.scales.size() == w.rows, "gemv: malformed QuantizedI8Matrix (scales size)");
  require(size_matches(w.values.size(), w.rows, w.row_stride),
          "gemv: malformed QuantizedI8Matrix (values size)");
}

void check_weights(const QuantizedQ4Matrix& w) {
  require(w.blocks_per_row == blocks_for(w.cols),
          "gemv: malformed QuantizedQ4Matrix (blocks_per_row)");
  require(size_matches(w.scales.size(), w.rows, w.blocks_per_row),
          "gemv: malformed QuantizedQ4Matrix (scales size)");
  require(
      size_matches(w.packed.size(), w.rows, w.blocks_per_row * QuantizedQ4Matrix::kBytesPerBlock),
      "gemv: malformed QuantizedQ4Matrix (packed size)");
}

// Shared GEMM driver. For every weight row in this thread's chunk, compute its dot product with
// all m activation rows before moving on, so the row (a few KiB) is streamed from memory once.
template <class RowDot>
void run_gemm(std::size_t n_rows, const std::vector<QuantizedActivations>& xs, std::span<float> y,
              const KernelOptions& opts, RowDot&& row_dot) {
  detail::run_rows(n_rows, opts.threads, opts.min_rows_per_thread,
                   [&](std::size_t begin, std::size_t end) {
                     for (std::size_t n = begin; n < end; ++n) {
                       for (std::size_t m = 0; m < xs.size(); ++m) {
                         y[m * n_rows + n] = row_dot(n, xs[m]);
                       }
                     }
                   });
}

std::vector<QuantizedActivations> quantize_rows(MatrixView x) {
  std::vector<QuantizedActivations> out;
  out.reserve(x.rows());
  for (std::size_t m = 0; m < x.rows(); ++m) {
    out.push_back(quantize_activations(x.row(m)));
  }
  return out;
}

}  // namespace

void gemv(const QuantizedI8Matrix& w, const QuantizedActivations& x, std::span<float> y,
          const KernelOptions& opts) {
  check_weights(w);
  check_activation(x, w.cols);
  require(y.size() == w.rows, "gemv: y.size() must equal weight rows");
  const DotI8Fn dot = detail::select_i8(opts.backend);
  const std::size_t blocks = x.blocks();
  detail::run_rows(
      w.rows, opts.threads, opts.min_rows_per_thread, [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
          y[r] = dot(w.row(r).data(), w.scales[r], x.values.data(), x.scales.data(), blocks);
        }
      });
}

void gemv(const QuantizedQ4Matrix& w, const QuantizedActivations& x, std::span<float> y,
          const KernelOptions& opts) {
  check_weights(w);
  check_activation(x, w.cols);
  require(y.size() == w.rows, "gemv: y.size() must equal weight rows");
  const DotQ4Fn dot = detail::select_q4(opts.backend);
  const std::size_t blocks = x.blocks();
  detail::run_rows(w.rows, opts.threads, opts.min_rows_per_thread,
                   [&](std::size_t begin, std::size_t end) {
                     for (std::size_t r = begin; r < end; ++r) {
                       y[r] = dot(w.row_packed(r).data(), w.row_scales(r).data(), x.values.data(),
                                  x.scales.data(), blocks);
                     }
                   });
}

void gemv(const QuantizedI8Matrix& w, std::span<const float> x, std::span<float> y,
          const KernelOptions& opts) {
  require(x.size() == w.cols, "gemv: x.size() must equal weight cols");
  gemv(w, quantize_activations(x), y, opts);
}

void gemv(const QuantizedQ4Matrix& w, std::span<const float> x, std::span<float> y,
          const KernelOptions& opts) {
  require(x.size() == w.cols, "gemv: x.size() must equal weight cols");
  gemv(w, quantize_activations(x), y, opts);
}

void gemm(const QuantizedI8Matrix& w, MatrixView x, std::span<float> y, const KernelOptions& opts) {
  check_weights(w);
  require(x.cols() == w.cols, "gemm: x.cols must equal weight cols");
  require(size_matches(y.size(), x.rows(), w.rows),
          "gemm: y.size() must equal x.rows * weight rows");
  const DotI8Fn dot = detail::select_i8(opts.backend);
  const auto xs = quantize_rows(x);
  const std::size_t blocks = blocks_for(w.cols);
  run_gemm(w.rows, xs, y, opts, [&](std::size_t n, const QuantizedActivations& a) {
    return dot(w.row(n).data(), w.scales[n], a.values.data(), a.scales.data(), blocks);
  });
}

void gemm(const QuantizedQ4Matrix& w, MatrixView x, std::span<float> y, const KernelOptions& opts) {
  check_weights(w);
  require(x.cols() == w.cols, "gemm: x.cols must equal weight cols");
  require(size_matches(y.size(), x.rows(), w.rows),
          "gemm: y.size() must equal x.rows * weight rows");
  const DotQ4Fn dot = detail::select_q4(opts.backend);
  const auto xs = quantize_rows(x);
  const std::size_t blocks = w.blocks_per_row;
  run_gemm(w.rows, xs, y, opts, [&](std::size_t n, const QuantizedActivations& a) {
    return dot(w.row_packed(n).data(), w.row_scales(n).data(), a.values.data(), a.scales.data(),
               blocks);
  });
}

}  // namespace int8k
