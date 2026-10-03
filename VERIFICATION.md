# int8k test record

- **Date**: 2026-10-03. Each preset uses a clean build tree.
- **Machine**: shared 4-vCPU Linux container (Ubuntu 24.04, Linux 6.18). `lscpu` reports
  an Intel Xeon @ 2.80 GHz with AVX2, FMA and AVX-512F, 4 MiB L2 (4 instances) and 33 MiB L3.
  Concurrent build jobs produce a load average of 7 to 20; timings are noisy (see Benchmarks).
- **Toolchain**: GCC 13.3.0, Clang 18.1.3, aarch64-linux-gnu-g++ 13.2 (Ubuntu cross package),
  qemu-aarch64 8.2.2, CMake 3.28, Ninja, GoogleTest 1.14.0 (apt `libgtest-dev`), clang-format
  18.1.3, actionlint 1.7.7.
- **Build storage**: successful results use `build/` symlinked to tmpfs (`/dev/shm`) and
  `TMPDIR` on tmpfs. Root-filesystem builds are FAIL: `No space left on device` in `portable`,
  `asan`, `tsan` and `arm64-cross` (other jobs leave about 10 to 80 MB free).
- **Deployment**: validated, never deployed; there are no users.

## Summary

| # | Command | Result | Key output |
|---|---|---|---|
| 1 | `cmake --preset release && cmake --build --preset release -j2` (GCC, `-Werror`) | PASS | 0 warnings |
| 2 | `ctest --preset release -j2` | PASS | `100% tests passed, 0 tests failed out of 259` (244 GoogleTest + 15 CLI/bench end-to-end), 0 skipped |
| 3 | `debug` preset: configure, build, `ctest` | PASS | 0 warnings; 259/259 |
| 4 | `clang-release` preset: configure, build, `ctest` | PASS | Clang 18.1.3, 0 warnings; 259/259 |
| 5 | `portable` preset (x86 kernels compiled out): configure, build, `ctest` | PASS | 259/259, 2 AVX2-only tests SKIPPED (`Gemv.Avx2MatchesScalar`, `ReferenceF32.Avx2MatchesScalarForAllTailLengths`); `nm` finds 0 `dot_*_avx2` symbols; `int8k-cli info` reports `avx2 available: no` |
| 6 | `asan` preset (`-fsanitize=address,undefined -fno-sanitize-recover=all`): configure, build, `ctest` | PASS | 257/257 (no bench binary in this preset, so no `bench.*` tests), no sanitizer reports; `int8k-cli selfcheck --rows 513 --cols 4100 --threads 4` also clean |
| 7 | `tsan` preset (`-fsanitize=thread`): configure, build, `ctest` | PASS | 257/257, no ThreadSanitizer warnings; the same selfcheck also clean |
| 8 | `arm64-cross` preset: cross-compile for aarch64 Linux with `-Werror`, GoogleTest built for aarch64 from `/usr/src/googletest`, every test run under `qemu-aarch64` | PASS | 0 warnings; `ELF 64-bit LSB pie executable, ARM aarch64`; 259/259 with the same 2 AVX2-only tests SKIPPED; `int8k-cli selfcheck --rows 513 --cols 4100 --threads 4` under QEMU: `selfcheck: PASS` |
| 9 | `clang-format --dry-run --Werror` on all 29 C++ files | PASS | clean |
| 10 | FetchContent fallback: `-DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON -DINT8K_FETCH_GTEST=ON -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/usr/src/googletest` | PASS | `GoogleTest not found; fetching v1.14.0 with FetchContent`; 259/259 |
| 11 | Configure without GoogleTest and without `INT8K_FETCH_GTEST` | PASS | expected configure error: `GoogleTest not found. Install it (e.g. sudo apt-get install libgtest-dev) or configure with -DINT8K_FETCH_GTEST=ON` |
| 12 | YAML parse of `.github/workflows/ci.yml` (PyYAML via `uv run --with pyyaml`) | PASS | `YAML OK, jobs: ['format', 'linux', 'sanitizers', 'arm64-cross', 'mutations', 'macos']` |
| 13 | `actionlint .github/workflows/ci.yml` | PASS | no findings |
| 14 | `TMPDIR=/dev/shm/int8k/tmp JOBS=2 tools/mutations/run_mutations.sh` (6 patches) | PASS | table below; exit 0 |
| 15 | `int8k-cli selfcheck --rows 513 --cols 4100 --threads 4` and `--rows 1 --cols 100 --threads 2` | PASS | `selfcheck: PASS` for both; worst error 0.17% to 0.33% of the derived tolerance |
| 16 | `int8k_tests --gtest_filter=Gemv.OracleRejectsZeroOutputThatTheLooseBoundAccepts` (loose-bound probe) | PASS | output below |
| 17 | CLI and bench input validation (`--threads -1`, `--threads abc`, `--reps 0` (int8k-bench only), `--rows 0`, `--stddev 1x`, `--format int3`, `--backend neon`, `--colz`, unknown command) | PASS | each rejected with exit code 2 and its specific message; 9 CTest tests check both via `cmake/ExpectFailure.cmake` |
| 18 | `int8k-bench --threads 4 --reps 15`, two runs back to back | PASS | tables below |
| 19 | Underflow probe (small program linking `libint8k.a`; source is not included): `w = 1e-20, x = 8.07e-22, K = 4096` and `w = x = 1e-19, K = 64` | PASS | numbers below; `Gemv.UnderflowingScaleProductsStayWithinAbsoluteBound` pins the same cases |
| 20 | GitHub Actions workflow actually executing on GitHub | NOT_RUN | No GitHub execution result is recorded; validation covers linting only (rows 12 and 13). |
| 21 | macOS / Apple silicon build | NOT_RUN | No macOS host in this Linux test environment. The portable code path is tested on x86 (row 5) and on aarch64 Linux under QEMU (row 8). |
| 22 | FetchContent download from github.com itself | NOT_RUN | github.com downloads are blocked in this test environment. Row 10 tests FetchContent with local GoogleTest sources only. |
| — | `portable`, `asan`, `tsan` and `arm64-cross` builds on the root filesystem | FAIL | `No space left on device`; about 10 to 80 MB free. The PASS results above use tmpfs. |

## Test inventory

```
$ ./build/release/tests/int8k_tests
[==========] 244 tests from 15 test suites ran.
[  PASSED  ] 244 tests.
```

- 64 unit and regression tests in `Cpu`, `Dispatch`, `Kernels`, `EffectiveThreads`,
  `ParallelForRows`, `RunRows`, `QuantizeI8`, `QuantizeQ4`, `QuantizeActivations`,
  `EdgeCases`, `Gemv`, `Gemm`, `ReferenceF32` and `Rng`.
- 180 parameterized property cases (`Property/RandomShapes`): 60 shapes (13 hand-picked
  boundary widths, 3 LLM widths: 4x4096, 3x11008, 2x4113, and 44 random shapes up to 80x700),
  each run as `Int8`, `Int4` and `Int4WithOutliers`. Each case compares scalar, AVX2 and a
  4-thread run with the exact oracle, with each other, and with the float64 product.
- 15 CTest-only end-to-end tests of the real binaries: `cli.info`, `cli.quantize_q4`,
  `cli.quantize_int8`, `cli.gemv`, `cli.selfcheck`, `cli.selfcheck_tiny`, seven `cli.rejects_*`
  and two `bench.rejects_*`. The rejection tests require exit code 2 **and** the expected
  message (`cmake/ExpectFailure.cmake`), so a crash or an unrelated error cannot pass.

## Mutation testing

Each mutation is a patch in `tools/mutations/`. `run_mutations.sh` copies the tree to a
temporary directory, applies one patch, builds the release configuration, and counts failures
in the GoogleTest binary and in CTest. The working tree is never modified.

| mutation | GoogleTest failures | CTest failures (of total) | caught |
|---|---|---|---|
| 1-swap-q4-nibble-halves-avx2 | 128 | 131 of 259 | yes |
| 2-truncate-instead-of-round | 5 | 5 of 259 | yes |
| 3-drop-odd-tail-block-int8-avx2 | 34 | 34 of 259 | yes |
| 4-drop-last-q4-block-in-driver | 127 | 129 of 259 | yes |
| 5-avx2-backend-dispatches-scalar | 2 | 2 of 259 | yes |
| 6-row-split-always-one-thread | 6 | 6 of 259 | yes |

| Patch | What it does | What catches it |
|---|---|---|
| 1 | swaps the low/high nibble halves in the AVX2 `unpack_q4` | the INT4 property cases, `Kernels.Q4*`, the CLI selfchecks |
| 2 | `std::trunc` instead of `std::nearbyint` in `quantize_one` | round-trip half-step tests, the subnormal bound test, one property case, `LargeFiniteResultsDoNotOverflow` |
| 3 | deletes the odd final block in `dot_i8_avx2` | 31 property cases (all with an odd block count), `Kernels.Int8DotIsExactOnExtremeAndRandomCodes`, `Gemv.Avx2MatchesScalar`, `Gemv.Int8ExactWhenQuantizationIsLossless` |
| 4 | passes `blocks - 1` for INT4 at both shared-driver call sites (`gemv` and `gemm`), so scalar and AVX2 are wrong in the same way | all 120 INT4 property cases plus `Gemv`/`Gemm` oracle tests |
| 5 | `select_i8`/`select_q4` return the scalar kernel for `Backend::Avx2` | `Dispatch.Avx2BackendSelectsAvx2Kernels`, `Dispatch.AutoSelectsTheFastestAvailableKernel` |
| 6 | `effective_threads` always returns 1 | five `EffectiveThreads.*` tests and `RunRows.UsesTheRequestedNumberOfOsThreads` |

## CLI output (release build)

```
$ int8k-cli info
int8k 0.1.0
avx2 available: yes
auto backend: avx2
hardware threads: 4

$ int8k-cli quantize --rows 4096 --cols 4096 --format int8
format: int8
shape: 4096x4096
f32_bytes: 67108864
quantized_bytes: 16793600
compression: 4.00x
bits_per_weight: 8.008
max_abs_error: 0.0224511
rmse: 0.00865585
signal_rms: 1.00004
snr_db: 41.25

$ int8k-cli quantize --rows 4096 --cols 4096 --format q4
format: q4
shape: 4096x4096
f32_bytes: 67108864
quantized_bytes: 10485760
compression: 6.40x
bits_per_weight: 5.000
max_abs_error: 0.403966
rmse: 0.0969818
signal_rms: 1.00004
snr_db: 20.27

$ int8k-cli selfcheck --rows 513 --cols 4100 --threads 4
int8  scalar threads=1   vs_oracle_max_abs=3.36e-06 (0.0031 of tolerance) snr_vs_f32_db=40.16 ok
int8  scalar threads=4   vs_oracle_max_abs=3.36e-06 (0.0031 of tolerance) snr_vs_f32_db=40.16 ok
int8  avx2   threads=1   vs_oracle_max_abs=2.07e-06 (0.002 of tolerance) snr_vs_f32_db=40.16 ok
int8  avx2   threads=4   vs_oracle_max_abs=2.07e-06 (0.002 of tolerance) snr_vs_f32_db=40.16 ok
q4    scalar threads=1   vs_oracle_max_abs=3.48e-06 (0.0033 of tolerance) snr_vs_f32_db=20.48 ok
q4    scalar threads=4   vs_oracle_max_abs=3.48e-06 (0.0033 of tolerance) snr_vs_f32_db=20.48 ok
q4    avx2   threads=1   vs_oracle_max_abs=1.82e-06 (0.0017 of tolerance) snr_vs_f32_db=20.48 ok
q4    avx2   threads=4   vs_oracle_max_abs=1.82e-06 (0.0017 of tolerance) snr_vs_f32_db=20.48 ok
selfcheck: PASS

$ int8k-cli gemv --rows 4096 --cols 11008 --format q4
format: q4
backend: avx2
threads: 1
shape: 4096x11008
weight_quantize_ms (offline step): 781.450
act_quantize+gemv_ms (single cold run, use int8k-bench for timing): 5.616
max_abs_error: 0.904061
rmse: 0.206153
signal_rms: 2.09522
snr_db: 20.14

$ int8k-cli gemv --rows 8 --cols 8 --threads -1 ; echo exit=$?
error: --threads must be an integer in [0, 1024], got '-1'
exit=2

$ int8k-bench --threads abc ; echo exit=$?
error: --threads must be an integer in [0, 1024], got 'abc'
usage: int8k-bench [--threads T] [--reps R] [--quick]
exit=2
```

`int8k-cli gemv` quantizes weights before starting the GEMV clock. Weight quantization is
an offline step; its time is reported separately.

## Loose bound vs exact oracle (row 16)

```
[probe] K=200: y=0 passes the loose bound for 12/64 INT8 and 61/64 INT4 rows, and the exact oracle for 0/64 INT8 and 0/64 INT4 rows
[probe] K=4096: y=0 passes the loose bound for 45/64 INT8 and 64/64 INT4 rows, and the exact oracle for 0/64 INT8 and 0/64 INT4 rows
[probe] K=11008: y=0 passes the loose bound for 63/64 INT8 and 64/64 INT4 rows, and the exact oracle for 0/64 INT8 and 1/64 INT4 rows
```

Weights N(0, 0.02²), activations N(0, 1), 64 rows. An all-zero output (a broken kernel) passes
the worst-case quantization bound for most rows, while the exact oracle accepts it only for one
INT4 row whose true result (0.0026) is itself within rounding tolerance (0.0029) of zero. The
test asserts that the oracle accepts zero for at most 2 of 64 rows.

## Underflow probe (row 19)

```
w=1e-20 x=8.07e-22 K=4096 scalar: int8 y=0 oracle=3.30547e-38 err=3.31e-38 tol=4.63e-38 | q4 y=3.06157e-38 err=2.44e-39 tol=4.63e-38
w=1e-20 x=8.07e-22 K=4096 avx2:   int8 y=0 oracle=3.30547e-38 err=3.31e-38 tol=4.63e-38 | q4 y=3.06157e-38 err=2.44e-39 tol=4.63e-38
w=1e-19 x=1e-19    K=64   scalar: int8 y=6.39352e-37 oracle=6.4e-37 err=6.48e-40 tol=7.24e-40 | q4 err=2.1e-41 tol=7.24e-40
w=1e-19 x=1e-19    K=64   avx2:   int8 y=6.39352e-37 oracle=6.4e-37 err=6.48e-40 tol=7.24e-40 | q4 err=2.1e-41 tol=7.24e-40
```

INT8 returns exactly 0 for the first case. `kernel_tolerance` in `src/reference.cpp` includes
an absolute term for subnormal intermediates (derivation in [DESIGN.md](DESIGN.md)); both
cases are within this tolerance.

## Benchmarks

Command: `./build/release/int8k-bench --threads 4 --reps 15`, two back-to-back runs on the
shared 4-vCPU Linux container described above.

Table metrics:

- **ms**: median of 15 repetitions after 2 warm-up calls.
- **GFLOP/s**: `2·rows·cols·batch / time`.
- **W GB/s**: weight bytes read divided by time.
- **vs f32**: speedup relative to the single-threaded AVX2 float32 GEMV/GEMM on the same shape.
- Quantized timings include activation quantization.

**Noise**: the host is shared with other jobs (load average 15 to 18 on 4 vCPUs during the
benchmark). Absolute times differ by up to about 2x between the two runs (for example f32 AVX2
4096x4096: 14.04 then 8.27 ms; INT8 AVX2 4096x4096: 1.13 then 2.18 ms), so quote speedup
ranges, not absolute latencies, and rerun on an idle machine (ideally with `taskset`) before
quoting a latency.

**Cache effects**: the 4096x4096 quantized weights (16.8 MB INT8, 10.5 MB INT4) are well under
the 33 MB L3, while the 64 MB float32 matrix is not, so that row is a best case. On the
11008-wide shapes the INT8 weights (45 MB) exceed L3 and the INT4 weights (22.5 MB of codes plus
5.6 MB of scales, about 28 MB) are close to its size. Calling those shapes "memory-bound" is an
inference from the byte counts; no `perf` counters or roofline measurement back it.

### Run 1

`uptime` before the run: load average `16.02, 18.47, 18.24`.

```
int8k-bench: avx2=yes hardware_threads=4 threads=4 reps=15 (median ms)
| shape        | kernel     | impl   | threads | batch |        ms |  GFLOP/s |  W GB/s |   vs f32 |
|--------------|------------|--------|---------|-------|-----------|----------|---------|----------|
| 4096x4096    | f32 gemv   | scalar |       1 |     1 |    38.579 |     0.87 |    1.74 |    0.36x |
| 4096x4096    | f32 gemv   | avx2   |       1 |     1 |    14.040 |     2.39 |    4.78 |    1.00x |
| 4096x4096    | int8 gemv  | scalar |       1 |     1 |     4.054 |     8.28 |    4.14 |    3.46x |
| 4096x4096    | q4 gemv    | scalar |       1 |     1 |     7.827 |     4.29 |    1.34 |    1.79x |
| 4096x4096    | int8 gemv  | scalar |       4 |     1 |     4.570 |     7.34 |    3.67 |    3.07x |
| 4096x4096    | q4 gemv    | scalar |       4 |     1 |     6.184 |     5.43 |    1.70 |    2.27x |
| 4096x4096    | int8 gemv  | avx2   |       1 |     1 |     1.127 |    29.76 |   14.90 |   12.45x |
| 4096x4096    | q4 gemv    | avx2   |       1 |     1 |     1.435 |    23.38 |    7.31 |    9.78x |
| 4096x4096    | int8 gemv  | avx2   |       4 |     1 |     4.035 |     8.32 |    4.16 |    3.48x |
| 4096x4096    | q4 gemv    | avx2   |       4 |     1 |     3.828 |     8.77 |    2.74 |    3.67x |
| 11008x4096   | f32 gemv   | scalar |       1 |     1 |    90.054 |     1.00 |    2.00 |    0.37x |
| 11008x4096   | f32 gemv   | avx2   |       1 |     1 |    33.060 |     2.73 |    5.46 |    1.00x |
| 11008x4096   | int8 gemv  | scalar |       1 |     1 |    18.268 |     4.94 |    2.47 |    1.81x |
| 11008x4096   | q4 gemv    | scalar |       1 |     1 |    29.195 |     3.09 |    0.97 |    1.13x |
| 11008x4096   | int8 gemv  | scalar |       4 |     1 |    17.301 |     5.21 |    2.61 |    1.91x |
| 11008x4096   | q4 gemv    | scalar |       4 |     1 |    36.803 |     2.45 |    0.77 |    0.90x |
| 11008x4096   | int8 gemv  | avx2   |       1 |     1 |     6.796 |    13.27 |    6.64 |    4.86x |
| 11008x4096   | q4 gemv    | avx2   |       1 |     1 |     3.591 |    25.11 |    7.85 |    9.21x |
| 11008x4096   | int8 gemv  | avx2   |       4 |     1 |     6.671 |    13.52 |    6.77 |    4.96x |
| 11008x4096   | q4 gemv    | avx2   |       4 |     1 |     4.254 |    21.20 |    6.63 |    7.77x |
| 4096x11008   | f32 gemv   | scalar |       1 |     1 |    89.144 |     1.01 |    2.02 |    0.39x |
| 4096x11008   | f32 gemv   | avx2   |       1 |     1 |    34.525 |     2.61 |    5.22 |    1.00x |
| 4096x11008   | int8 gemv  | scalar |       1 |     1 |    16.614 |     5.43 |    2.71 |    2.08x |
| 4096x11008   | q4 gemv    | scalar |       1 |     1 |    32.550 |     2.77 |    0.87 |    1.06x |
| 4096x11008   | int8 gemv  | scalar |       4 |     1 |    19.642 |     4.59 |    2.30 |    1.76x |
| 4096x11008   | q4 gemv    | scalar |       4 |     1 |    30.075 |     3.00 |    0.94 |    1.15x |
| 4096x11008   | int8 gemv  | avx2   |       1 |     1 |     9.644 |     9.35 |    4.68 |    3.58x |
| 4096x11008   | q4 gemv    | avx2   |       1 |     1 |     5.794 |    15.56 |    4.86 |    5.96x |
| 4096x11008   | int8 gemv  | avx2   |       4 |     1 |     7.717 |    11.69 |    5.84 |    4.47x |
| 4096x11008   | q4 gemv    | avx2   |       4 |     1 |     4.433 |    20.34 |    6.36 |    7.79x |
| 4096x4096    | f32 gemm   | avx2   |       1 |     8 |    70.089 |     3.83 |    0.96 |    1.00x |
| 4096x4096    | int8 gemm  | avx2   |       1 |     8 |    11.124 |    24.13 |    1.51 |    6.30x |
| 4096x4096    | q4 gemm    | avx2   |       1 |     8 |    13.586 |    19.76 |    0.77 |    5.16x |
| 4096x4096    | int8 gemm  | avx2   |       4 |     8 |    13.625 |    19.70 |    1.23 |    5.14x |
| 4096x4096    | q4 gemm    | avx2   |       4 |     8 |    12.533 |    21.42 |    0.84 |    5.59x |
```

### Run 2

`uptime` before the run: load average `15.02, 18.08, 18.12`.

```
int8k-bench: avx2=yes hardware_threads=4 threads=4 reps=15 (median ms)
| shape        | kernel     | impl   | threads | batch |        ms |  GFLOP/s |  W GB/s |   vs f32 |
|--------------|------------|--------|---------|-------|-----------|----------|---------|----------|
| 4096x4096    | f32 gemv   | scalar |       1 |     1 |    23.687 |     1.42 |    2.83 |    0.35x |
| 4096x4096    | f32 gemv   | avx2   |       1 |     1 |     8.266 |     4.06 |    8.12 |    1.00x |
| 4096x4096    | int8 gemv  | scalar |       1 |     1 |     3.688 |     9.10 |    4.55 |    2.24x |
| 4096x4096    | q4 gemv    | scalar |       1 |     1 |     9.950 |     3.37 |    1.05 |    0.83x |
| 4096x4096    | int8 gemv  | scalar |       4 |     1 |     6.763 |     4.96 |    2.48 |    1.22x |
| 4096x4096    | q4 gemv    | scalar |       4 |     1 |    10.129 |     3.31 |    1.04 |    0.82x |
| 4096x4096    | int8 gemv  | avx2   |       1 |     1 |     2.178 |    15.41 |    7.71 |    3.80x |
| 4096x4096    | q4 gemv    | avx2   |       1 |     1 |     1.023 |    32.78 |   10.25 |    8.08x |
| 4096x4096    | int8 gemv  | avx2   |       4 |     1 |     4.017 |     8.35 |    4.18 |    2.06x |
| 4096x4096    | q4 gemv    | avx2   |       4 |     1 |     3.997 |     8.39 |    2.62 |    2.07x |
| 11008x4096   | f32 gemv   | scalar |       1 |     1 |    71.260 |     1.27 |    2.53 |    0.35x |
| 11008x4096   | f32 gemv   | avx2   |       1 |     1 |    25.191 |     3.58 |    7.16 |    1.00x |
| 11008x4096   | int8 gemv  | scalar |       1 |     1 |    13.378 |     6.74 |    3.37 |    1.88x |
| 11008x4096   | q4 gemv    | scalar |       1 |     1 |    25.256 |     3.57 |    1.12 |    1.00x |
| 11008x4096   | int8 gemv  | scalar |       4 |     1 |    11.148 |     8.09 |    4.05 |    2.26x |
| 11008x4096   | q4 gemv    | scalar |       4 |     1 |    45.928 |     1.96 |    0.61 |    0.55x |
| 11008x4096   | int8 gemv  | avx2   |       1 |     1 |    10.815 |     8.34 |    4.17 |    2.33x |
| 11008x4096   | q4 gemv    | avx2   |       1 |     1 |     3.860 |    23.36 |    7.30 |    6.53x |
| 11008x4096   | int8 gemv  | avx2   |       4 |     1 |     7.501 |    12.02 |    6.02 |    3.36x |
| 11008x4096   | q4 gemv    | avx2   |       4 |     1 |     3.996 |    22.57 |    7.05 |    6.30x |
| 4096x11008   | f32 gemv   | scalar |       1 |     1 |    67.548 |     1.34 |    2.67 |    0.36x |
| 4096x11008   | f32 gemv   | avx2   |       1 |     1 |    24.083 |     3.74 |    7.49 |    1.00x |
| 4096x11008   | int8 gemv  | scalar |       1 |     1 |    15.672 |     5.75 |    2.88 |    1.54x |
| 4096x11008   | q4 gemv    | scalar |       1 |     1 |    32.642 |     2.76 |    0.86 |    0.74x |
| 4096x11008   | int8 gemv  | scalar |       4 |     1 |    15.391 |     5.86 |    2.93 |    1.56x |
| 4096x11008   | q4 gemv    | scalar |       4 |     1 |    30.573 |     2.95 |    0.92 |    0.79x |
| 4096x11008   | int8 gemv  | avx2   |       1 |     1 |     7.068 |    12.76 |    6.38 |    3.41x |
| 4096x11008   | q4 gemv    | avx2   |       1 |     1 |     5.096 |    17.70 |    5.53 |    4.73x |
| 4096x11008   | int8 gemv  | avx2   |       4 |     1 |     9.563 |     9.43 |    4.72 |    2.52x |
| 4096x11008   | q4 gemv    | avx2   |       4 |     1 |     6.121 |    14.73 |    4.60 |    3.93x |
| 4096x4096    | f32 gemm   | avx2   |       1 |     8 |    71.319 |     3.76 |    0.94 |    1.00x |
| 4096x4096    | int8 gemm  | avx2   |       1 |     8 |     6.078 |    44.16 |    2.76 |   11.73x |
| 4096x4096    | q4 gemm    | avx2   |       1 |     8 |     9.578 |    28.03 |    1.09 |    7.45x |
| 4096x4096    | int8 gemm  | avx2   |       4 |     8 |    11.677 |    22.99 |    1.44 |    6.11x |
| 4096x4096    | q4 gemm    | avx2   |       4 |     8 |     6.869 |    39.08 |    1.53 |   10.38x |
```

### Derived ratios (from the two runs above)

| Metric | Run 1 | Run 2 |
|---|---|---|
| INT8 AVX2 vs f32 AVX2, 11008x4096 / 4096x11008 | 4.86x / 3.58x | 2.33x / 3.41x |
| INT4 AVX2 vs f32 AVX2, 11008x4096 / 4096x11008 | 9.21x / 5.96x | 6.53x / 4.73x |
| INT8 / INT4 AVX2 vs f32 AVX2, 4096x4096 (L3-resident, best case) | 12.46x / 9.78x | 3.80x / 8.08x |
| INT4 AVX2 vs INT4 scalar, 11008x4096 / 4096x11008 | 8.13x / 5.62x | 6.54x / 6.41x |
| INT8 AVX2 vs INT8 scalar, 11008x4096 / 4096x11008 | 2.69x / 1.72x | 1.24x / 2.22x |
| f32 AVX2 vs f32 strict scalar (all three shapes) | 2.58x to 2.75x | 2.80x to 2.87x |
| 4 threads vs 1 thread, AVX2 kernels | 0.28x to 1.31x | 0.26x to 1.44x |
| 4 threads vs 1 thread, scalar kernels | 0.79x to 1.27x | 0.55x to 1.20x |
| INT8 / INT4 GEMM batch 8, 4096x4096, 1 thread | 11.12 / 13.59 ms (f32: 70.09 ms) | 6.08 / 9.58 ms (f32: 71.32 ms) |

Over both runs on the 11008-wide shapes: INT4 AVX2 GEMV is 4.7x to 9.2x and INT8 AVX2 GEMV
2.3x to 4.9x faster than the AVX2 float32 GEMV. The 12.46x INT8 figure for 4096x4096 in run 1
coincides with a slow float32 run (14.04 ms vs 8.27 ms in run 2) and should not be quoted.

The threading results are not a reliable measure of scaling. The container is heavily shared,
threads are spawned per call, and GEMV streams a lot of memory per flop. Rerun on an idle
machine before quoting any threaded number.
