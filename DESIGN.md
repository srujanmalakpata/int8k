# int8k design

## Goals and non-goals

**Goals**:

- A small set of quantized kernels that are carefully tested: every backend is checked against
  an exact oracle within a derived rounding tolerance, against the scalar kernel, and against
  the unquantized product within the quantization error bound.
- Measurably faster than a fair float32 baseline on LLM-shaped layers.
- Portable: one x86 binary runs with or without AVX2, and the same source compiles to the
  portable kernels elsewhere. The recorded aarch64 Linux cross-build passes the test suite
  under QEMU; the macOS job is configured in CI but has not run on GitHub.

**Non-goals**: a model loader, a tokenizer, a full inference engine, or competing with
llama.cpp, oneDNN or BLAS on peak throughput.

## Key decisions

### 1. Symmetric quantization with the range [-127, 127] (and [-7, 7] for INT4)

The scale is `max|x| / 127` and the code is `round(x / scale)`, clamped.

- **Why symmetric**: weights and activations in transformer layers are roughly zero-centred.
  Symmetric quantization needs no zero-point, so the integer dot product needs no correction
  terms (asymmetric quantization adds `zp * Σx`-style terms per block).
- **Why -128 is never used**:
  - Negation stays exact.
  - The error bound is simply `|x - x̂| ≤ scale / 2` for normal inputs. The exception is a
    group whose `amax / 127` is subnormal: the scale then has few significant bits and is
    rounded down, so the largest values clamp to ±127 and the bound becomes
    `max(scale / 2, amax − 127·scale)`. When the scale underflows to 0 every value becomes 0,
    an error of up to `amax` (this happens for INT8 when `amax < 127 × denorm_min ≈ 1.8e-43`). `SubnormalInputsStayWithinDocumentedBound`
    asserts this bound.
  - The AVX2 trick in decision 4 relies on `|x| ≤ 127`, which keeps pairwise int16 sums below
    the saturation limit.

  The cost is giving up one of 256 levels. `quantize_i8` never produces -128. A hand-built
  `QuantizedI8Matrix` with a -128 weight code is a precondition violation: the sign trick
  computes `-(-128)` as -128 in 8 bits, so AVX2 and scalar disagree. Scanning every code on
  every call would add a full pass over the weights, so only buffer sizes are validated
  (decision 10).
- **INT4 range**: [-7, 7] is used for the same reasons. llama.cpp's Q4_0 instead uses
  `d = signed_max / -8` so the extreme value maps to -8, which gains one level for one sign.
  The symmetric range gives a simpler, provable bound. The `QuantizeQ4` tests and
  `gemv_error_bound` rely on it.

### 2. Granularity: per-row for INT8, per-32-block for INT4, per-32-block for activations

- **INT8 per output row** (per-channel): 127 levels are enough that one scale per row loses
  little (41 dB SNR on Gaussian data). Mathematically the row scale factors out of the dot
  product, but the kernel still multiplies it into every block's scale (`ws * xs[b]`, as INT4
  does). Applying it once at the end allows the intermediate `Σ_b xs[b] * dot_b` to overflow
  to `inf` for `x = 1e37`, `w = 1e-3`, although the true result is `6.4e35`.
  `Gemv.LargeFiniteResultsDoNotOverflow` checks this case. The extra multiply per 32 elements
  has no measured speed impact in an interleaved A/B timing.

  **The trade-off**: folding moves the failure from overflow to underflow. When
  `max|w_row| · max|x_block|` is below about 2e-34 (INT8; about 1e-35 for INT4), the scale
  product `ws · xs[b]` drops below `FLT_MIN` and becomes subnormal or 0, so the result keeps
  only an absolute accuracy of about `blocks × 3.6e-40` and can lose all relative precision.
  For example `w = 1e-20`, `x = 8.07e-22`, K = 4096 returns exactly 0 for INT8 although the true
  result is 3.3e-38. Overflow gave `inf` for realistic-looking magnitudes (activation outliers
  of 1e37 are rare but finite); underflow only affects results below about 1e-30, far under
  anything a real layer produces, so this is the better side to fail on. `kernel_tolerance`
  carries an absolute term sized for it, and `Gemv.UnderflowingScaleProductsStayWithinAbsoluteBound`
  pins the behaviour.
- **INT4 per 32-element block**: with only 15 levels, one outlier per row would crush every
  other value to zero. A scale per 32 values confines each outlier to its block. The test
  `EachBlockHasItsOwnScale` and the `Int4WithOutliers` property tests check this.
- **Activations per 32-element block**: activations are not known ahead of time and have
  outliers (a known problem in LLMs), so they get block scales too. Using the same block size
  for both operands means the block dot products line up: `y = Σ_b s_w[b] * s_x[b] * dot_b`.

### 3. Pad rows to a multiple of 32 with zeros

Every quantized row has a whole number of blocks, so no kernel needs a scalar tail loop. A
zero code contributes nothing to a dot product, so padding is free in terms of correctness. The
cost is up to 31 bytes per row, which is negligible for real layer widths (multiples of 128).
Tests cover widths of 1, 15, 17, 31, 33, 63, 65 and random widths.

### 4. AVX2 signed x signed int8 dot product

AVX2 has no instruction that multiplies signed bytes by signed bytes and adds the results.
`_mm256_maddubs_epi16` multiplies *unsigned* by *signed* bytes. The kernel therefore applies the
identity `x * w = |x| * (w * sign(x))`:

1. `_mm256_sign_epi8(x, x)` computes |x|.
2. `_mm256_sign_epi8(w, x)` moves the sign of x onto w.
3. `maddubs` produces 16 int16 pairwise sums.
4. `_mm256_madd_epi16` with a vector of ones widens those to 8 int32 sums.

Each block's 8 partial sums are converted to float and accumulated with FMA into two
accumulators. Two chains do not fully hide FMA latency (about 4 cycles at 2 FMAs per cycle
would need about 8 chains), but that does not matter here: the per-block integer work
(`sign`, `maddubs`, `madd`, convert) and memory traffic dominate, not the FMA chain. For INT4, 16 packed bytes are unpacked with
`and 0x0F`, `srli 4` and `inserti128`, then 8 is subtracted.

**Packed INT4 layout**: byte j holds element j in its low nibble and element j+16 in its high
nibble. This differs from "two adjacent elements per byte" so that a single 128-bit mask/shift
produces elements 0..15 and 16..31 already in order. llama.cpp uses the same layout trick.

### 5. Runtime dispatch using `__attribute__((target("avx2,fma")))`

The alternative is compiling `kernels_avx2.cpp` with `-mavx2`. That is a known trap:

- Inline functions from headers (`std::vector`, `std::span` and so on) instantiated in that
  translation unit are also compiled with AVX2.
- The linker keeps one copy of each such function and may keep the AVX2 one.
- The binary then crashes with SIGILL on CPUs without AVX2.

The target attribute restricts AVX2 to the annotated functions. Dispatch happens once per call:
`detail::select_i8` and `detail::select_q4` return a function pointer, so no per-row branch is
needed. They are declared in `src/kernels.hpp` so `tests/test_kernels.cpp` can assert that
`Backend::Avx2` maps to `dot_*_avx2`. A dispatcher that silently returns the scalar kernel
turns every "AVX2 vs scalar" check into a scalar-vs-scalar comparison. The dispatch tests
detect this fault (mutation 5).
`cpu_has_avx2()` caches the result of `__builtin_cpu_supports`, which also checks that the OS
saves YMM state. The `INT8K_PORTABLE` CMake option compiles the x86 path out, so the
ARM/macOS code path can be tested on an x86 machine (the `portable` preset, run locally and
configured as a CI job).

### 6. Activations are quantized inside `gemv` (and timed)

A decode step must quantize its input vector before the integer kernel can run, so leaving
quantization out of the benchmark would flatter the results. Activation quantization is O(K),
while GEMV is O(N·K), so a scalar implementation is fine. An overload taking
`QuantizedActivations` lets callers reuse one quantized vector across, for example, the Q, K and
V projections.

### 7. Threading: static contiguous row split, one writer per output

- Each thread owns a contiguous range of output rows, so there are no shared writes, no atomics
  and no false sharing except at chunk edges.
- Results are bit-identical to single-threaded results because each row's arithmetic is
  unchanged.
- Exceptions thrown inside workers are captured and rethrown after all threads have joined.
- `min_rows_per_thread` (default 64) avoids spawning threads for tiny problems, and the count is
  capped at 4x the hardware threads so a huge request (for example a wrapped `-1`) cannot
  exhaust the OS thread limit.
- If creating a thread throws `std::system_error`, the workers already started are joined
  before the error propagates. Destroying a joinable `std::thread` would call `std::terminate`.
  (`std::jthread` would do this automatically, but older libc++ on macOS lacks it.)
- The `tsan` preset runs the whole suite under ThreadSanitizer.
- `gemv` and `gemm` both go through `detail::run_rows` (effective thread count, then the
  split). `tests/test_parallel.cpp` checks the count rules, that 4 requested threads run 4
  chunks on 4 distinct OS threads, and that an exception from a worker or from the caller's own
  chunk is rethrown after the join. An `effective_threads` that always returns 1 still gives
  numerically correct outputs and passes race checks; the thread-count tests detect this
  fault (mutation 6).

**Trade-off**: threads are spawned on every call. That is simple and has no global state, but
costs tens of microseconds per call. A sub-millisecond GEMV notices this, and the benchmark
shows it.

### 8. NaN policy: reject

`quantize_*` throws `std::invalid_argument` on NaN or Inf. A single NaN would set the block's
scale to NaN (or be silently skipped by `max`, depending on operand order) and corrupt 32 or
more outputs in a hard-to-debug way. Rejecting it is O(n) and fused into the max-abs pass that
already runs. The alternatives were:

- propagate NaN to the affected outputs, which is what float GEMV does;
- clamp Inf to the largest finite value.

Either could be a flag later.

### 9. Overflow-safe scales

For `max|x| = FLT_MAX`, `FLT_MAX / 127` rounds *up*, so dequantizing `127 * scale` can overflow
to +Inf. `symmetric_scale` steps the scale down with `nextafter` until `code * scale ≤ amax`.
`EdgeCases.MaxMagnitudeValuesDoNotOverflow` checks this case. Normally this takes zero steps,
and in the worst case one or two. The quantizer also divides (`x / scale`) instead of
multiplying by `1/scale`, because `1/scale` overflows for subnormal scales.

### 10. Validate hand-built structs, cheaply

The formats are public aggregates so they are easy to inspect, serialize and construct in
tests. The kernels read raw pointers, so `gemv` and `gemm` check every buffer size against
`rows`, `cols`, `row_stride` and `blocks_per_row` (O(1)) before any kernel runs, and throw
`std::invalid_argument` on a mismatch. A 4-row matrix with one row of codes would otherwise
cause a heap-buffer-overflow; `Gemv.MalformedWeightsAreRejected` checks rejection. Making the
formats classes with private data built only by `quantize_*` would be stricter; see Planned
improvements.

`MatrixView` already works that way: its fields are private behind `rows()`, `cols()` and
`data()`, so its `size == rows · cols` invariant cannot be broken after construction, and the
constructor checks it without computing the product: `2^32 × 2^32` wraps to 0 in 64-bit
arithmetic. `EdgeCases.MatrixViewRejectsOverflowingShape` checks rejection of that shape on
an empty buffer. `gemm` checks `y.size()` the same way.

## Testing strategy

- **Exact oracle**: `gemv_dequantized_f64` computes `T = Σ ŵ·x̂` in double from
  `dequantize()`, which decodes the nibble layout and scale indexing with code that is separate
  from the kernels. Every backend must satisfy `|y − T| ≤ γ(nb + 8)·Σ|ŵx̂|`, where
  `γ(n) = n·u / (1 − n·u)`, `u = 2^-24` and `nb` is the number of 32-element blocks.
  Derivation: each block's integer dot is exact (|dot| ≤ 516,128 < 2^24, also exact as a
  float). One term then passes through at most `nb + 5` float roundings: the scale product
  `ws·xs`, the multiply (scalar) or FMA, the sequential accumulation (`nb − 1` adds for scalar,
  `ceil(nb/2)` FMAs per AVX2 accumulator), the `acc0 + acc1` add and the 3-level horizontal sum.
  The oracle adds 2 roundings (`dequantize` of w and of x); its double products are exact and
  its double sum is accurate to about `K·2^-53`. Standard error analysis then gives the bound,
  with 1 spare rounding. A float operation whose result is subnormal has an absolute error of
  up to `denorm_min / 2` instead of a relative one, so the tolerance adds
  `denorm_min/2 · (nb · (516,128 + 4) + 8)`: the scale product's absolute error times the
  largest block dot, plus one more for every other operation. That term is about `nb × 3.6e-40`,
  irrelevant for real data. The bound assumes the stored scales are normal floats or 0 (group
  maxima above about 1.5e-36), so that the oracle's own dequantized values have only relative
  rounding errors. The bound is worst case, so measured errors are about 0.2% of it
  (`int8k-cli selfcheck`), yet a missing block, a wrong scale index or a nibble swap moves the
  result by far more. This is what catches bugs shared by both backends: dropping the last INT4
  block at both shared-driver call sites (mutation 4) fails 127 GoogleTest cases.
- **Quantization bound**: for `ŵ = w + δw` and `x̂ = x + δx`,
  `|ŵ·x̂ − w·x| ≤ Σ(|δw||x| + |w||δx| + |δw||δx|) + γ(nb + 8)·Σ|ŵx̂|`, with each `|δ|` at most
  half the step of its own scale (plus a few ulps for the rounded division). This checks that
  results are as accurate as the quantization step allows. It is a worst-case bound (every
  error assumed to have the same sign), so it is often larger than `|y|` itself. The test
  `Gemv.OracleRejectsZeroOutputThatTheLooseBoundAccepts` measures this on 64 rows of
  N(0, 0.02²) weights and N(0, 1) activations: an all-zero output passes the INT4 bound for
  61, 64 and 64 of 64 rows at K = 200, 4096 and 11008, and the INT8 bound for 12, 45 and 63
  rows, while the exact oracle accepts it for at most 1 row (one whose true value is itself
  within rounding tolerance of 0). That is why the loose bound is not used as the bug detector.
- **Cross-backend checks**: AVX2 and scalar are each within the oracle tolerance, so they must
  agree within twice that.
- **Threading checks**: exact equality, which is a strong guarantee and would catch an
  off-by-one in the chunking. Races are checked separately with ThreadSanitizer. Exact equality
  would also hold if nothing ran in parallel, so `test_parallel.cpp` checks the splitter itself
  (thread counts, distinct OS thread ids, exception propagation).
- **Kernel and dispatch checks**: `test_kernels.cpp` calls each kernel directly on hand-built
  blocks with known exact results, and asserts which kernel each `Backend` selects, so a
  dispatcher bug cannot turn the "AVX2 vs scalar" comparisons into scalar vs scalar.
- **Property-style tests**: 60 shapes (13 hand-picked boundary widths, 3 LLM widths: 4x4096,
  3x11008 and 2x4113, and 44 random shapes up to 80x700), with weight scales from 1e-4 to 1e2
  and activation scales from 1e-3 to 1e3, run through `TEST_P` so each failure names its
  shape. The generator uses SplitMix64 plus Box-Muller, because
  `std::normal_distribution` differs between libstdc++ and libc++ and CI on macOS would
  otherwise see different data.
- **Mutation testing**: six deliberate bugs are stored as patches in `tools/mutations/`, and
  `run_mutations.sh` applies each to a temporary copy, builds it and counts failures (a CI job
  runs it too). They swap the nibble halves (128 GoogleTest failures), truncate instead of
  rounding (5), drop the odd tail block in AVX2 (34), drop the last INT4 block in the shared
  driver (127), make the AVX2 backend dispatch to the scalar kernel (2) and make the row split
  always use one thread (6). See [VERIFICATION.md](VERIFICATION.md).

## Alternatives considered

| Option | Why not (for now) |
|---|---|
| Google Benchmark | Adds a dependency. The benchmark uses a 60-line median-of-N timing loop for these comparisons. |
| fp16 scales (as in llama.cpp) | Would give 4.5 instead of 5.0 bits per weight. It needs F16C conversion or a portable half type, which is more surface area. It is the first format change to make. |
| AVX-512 VNNI / AVX-VNNI (`vpdpbusd`) | Fuses maddubs and madd into one instruction, about 2x the integer throughput. Not every CI runner has it, and on large shapes the GEMV is probably limited by memory traffic rather than integer throughput (an inference from byte counts, not yet measured with `perf`). |
| OpenMP for threading | One pragma, but adds a toolchain dependency (AppleClang ships without libomp). `std::thread` keeps the build dependency-free. |
| Dequantize to float, then float GEMV | Simple, but loses the integer-throughput advantage and needs a float buffer per row. |
| Asymmetric (zero-point) quantization | Better for skewed distributions such as post-ReLU activations, but adds correction terms. Transformer weights are roughly symmetric. |

## Planned improvements

1. A persistent thread pool with a spin-then-sleep barrier, to remove the per-call spawn cost
   that dominates sub-millisecond GEMVs.
2. AVX-VNNI and AVX-512 VNNI kernels plus a NEON (`vdotq_s32`) kernel, behind the same
   dispatch.
3. fp16 scales and a llama.cpp-compatible Q4_0/Q8_0 layout, so real GGUF weights can be loaded
   and accuracy measured on a real model.
4. A register-blocked GEMM microkernel (for example 4 rows by 4 batch columns) for prefill.
5. Run benchmarks on a dedicated machine with `perf stat`, and report achieved DRAM bandwidth
   against the STREAM bandwidth of the same box (a roofline). That would confirm or refute the
   "memory-bound" reading of the 11008-wide results.
6. Make the quantized formats classes with private data, constructed only by `quantize_*` or a
   validating constructor, so the -128 precondition cannot be violated.
