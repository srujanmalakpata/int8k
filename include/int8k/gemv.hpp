// Quantized matrix-vector (GEMV) and small matrix-matrix (GEMM) products.
//
// Conventions follow a linear layer y = W x, with W stored as [out_features x in_features]
// (rows = outputs, cols = inputs), as in PyTorch's nn.Linear.
//
// Activations are quantized on the fly to block-wise INT8 (see QuantizedActivations), the inner
// products run in exact integer arithmetic per 32-element block, and each block's int32 result is
// multiplied by (weight scale * activation scale) and accumulated in float32.
//
// Every function validates shapes and the buffer sizes of the quantized operands and throws
// std::invalid_argument on a mismatch, before any kernel reads memory.
#pragma once

#include <cstddef>
#include <span>

#include "int8k/cpu.hpp"
#include "int8k/matrix.hpp"
#include "int8k/quantize.hpp"

namespace int8k {

struct KernelOptions {
  Backend backend = Backend::Auto;
  /// Worker threads for the row split. 1 = run on the calling thread, 0 = one per hardware
  /// thread. The effective count is capped at 4x the hardware threads, and so that every thread
  /// gets at least `min_rows_per_thread` rows.
  unsigned threads = 1;
  std::size_t min_rows_per_thread = 64;
};

/// y = W x for INT8 per-row weights. Requires x.size() == w.cols and y.size() == w.rows.
void gemv(const QuantizedI8Matrix& w, std::span<const float> x, std::span<float> y,
          const KernelOptions& opts = {});
/// y = W x for block-wise INT4 weights.
void gemv(const QuantizedQ4Matrix& w, std::span<const float> x, std::span<float> y,
          const KernelOptions& opts = {});

/// Variants that take already-quantized activations (reuse across several weight matrices).
void gemv(const QuantizedI8Matrix& w, const QuantizedActivations& x, std::span<float> y,
          const KernelOptions& opts = {});
void gemv(const QuantizedQ4Matrix& w, const QuantizedActivations& x, std::span<float> y,
          const KernelOptions& opts = {});

/// Y = X W^T: X is [m x w.cols] (a batch of m activation rows), Y is [m x w.rows] row-major.
/// Each weight row is loaded once and reused for all m activation rows while it is hot in cache.
void gemm(const QuantizedI8Matrix& w, MatrixView x, std::span<float> y,
          const KernelOptions& opts = {});
void gemm(const QuantizedQ4Matrix& w, MatrixView x, std::span<float> y,
          const KernelOptions& opts = {});

}  // namespace int8k
