// Internal: split a row range across std::threads.
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

namespace int8k::detail {

/// Upper bound on worker threads per hardware thread. More than this only adds spawn cost and
/// contention, and an unbounded request (e.g. a wrapped "-1") could exhaust the OS thread limit.
inline constexpr unsigned kMaxThreadsPerCore = 4;

/// Number of threads to actually use for `rows` rows: `requested` (0 = one per hardware thread),
/// capped at kMaxThreadsPerCore * hardware threads and so that every thread gets at least
/// `min_rows_per_thread` rows.
inline unsigned effective_threads(std::size_t rows, unsigned requested,
                                  std::size_t min_rows_per_thread) {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  unsigned t = requested == 0 ? hw : std::min(requested, hw * kMaxThreadsPerCore);
  const std::size_t by_work =
      std::max<std::size_t>(1, rows / std::max<std::size_t>(1, min_rows_per_thread));
  t = static_cast<unsigned>(std::min<std::size_t>(t, by_work));
  return std::max(1u, t);
}

/// Calls fn(begin, end) on `threads` contiguous, near-equal chunks of [0, rows). Chunk i runs on
/// worker thread i (chunk 0 on the calling thread). Each output row is written by exactly one
/// thread, so no synchronization is needed beyond the final join. The first exception thrown by
/// any chunk is rethrown on the calling thread after all workers have joined. If creating a
/// thread fails (std::system_error), the workers already started are joined before the error
/// propagates, so no joinable std::thread is ever destroyed (which would call std::terminate).
template <class Fn>
void parallel_for_rows(std::size_t rows, unsigned threads, Fn&& fn) {
  if (threads <= 1 || rows == 0) {
    fn(std::size_t{0}, rows);
    return;
  }
  const std::size_t base = rows / threads;
  const std::size_t extra = rows % threads;
  auto chunk_begin = [&](std::size_t i) { return i * base + std::min(i, extra); };

  std::vector<std::exception_ptr> errors(threads);
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  auto join_all = [&workers] {
    for (auto& t : workers) {
      if (t.joinable()) t.join();
    }
  };
  try {
    for (unsigned i = 1; i < threads; ++i) {
      workers.emplace_back([&, i] {
        try {
          fn(chunk_begin(i), chunk_begin(i + 1));
        } catch (...) {
          errors[i] = std::current_exception();
        }
      });
    }
  } catch (...) {
    join_all();
    throw;
  }
  try {
    fn(chunk_begin(0), chunk_begin(1));
  } catch (...) {
    errors[0] = std::current_exception();
  }
  join_all();
  for (const auto& e : errors) {
    if (e) std::rethrow_exception(e);
  }
}

/// What gemv and gemm call: parallel_for_rows with effective_threads(rows, requested,
/// min_rows_per_thread) threads.
template <class Fn>
void run_rows(std::size_t rows, unsigned requested, std::size_t min_rows_per_thread, Fn&& fn) {
  parallel_for_rows(rows, effective_threads(rows, requested, min_rows_per_thread),
                    std::forward<Fn>(fn));
}

}  // namespace int8k::detail
