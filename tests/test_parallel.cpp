// Tests for the row splitter in src/parallel.hpp.
//
// "Threaded output is bit-identical to single-threaded" (test_gemv.cpp) would also pass if the
// splitter quietly ran everything on one thread, and ThreadSanitizer would then have no
// concurrency to check. These tests pin the thread count, the chunking, which OS thread runs
// each chunk, and exception propagation.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "parallel.hpp"

using int8k::detail::effective_threads;
using int8k::detail::kMaxThreadsPerCore;
using int8k::detail::parallel_for_rows;
using int8k::detail::run_rows;

namespace {

unsigned hw_threads() {
  return std::max(1u, std::thread::hardware_concurrency());
}

struct Chunk {
  std::size_t begin;
  std::size_t end;
  std::thread::id thread;
};

// Runs `run(fn)` with an fn that records every chunk it is called with.
template <class Run>
std::vector<Chunk> record_chunks(Run&& run) {
  std::mutex mutex;
  std::vector<Chunk> chunks;
  run([&](std::size_t begin, std::size_t end) {
    const std::lock_guard<std::mutex> lock(mutex);
    chunks.push_back({begin, end, std::this_thread::get_id()});
  });
  std::sort(chunks.begin(), chunks.end(),
            [](const Chunk& a, const Chunk& b) { return a.begin < b.begin; });
  return chunks;
}

}  // namespace

TEST(EffectiveThreads, OneMeansCallingThreadOnly) {
  EXPECT_EQ(effective_threads(1000, 1, 1), 1u);
}

TEST(EffectiveThreads, ZeroMeansOnePerHardwareThread) {
  EXPECT_EQ(effective_threads(1'000'000, 0, 1), hw_threads());
}

TEST(EffectiveThreads, ExplicitCountIsHonouredWhenThereIsEnoughWork) {
  EXPECT_EQ(effective_threads(257, 4, 1), 4u);
  EXPECT_EQ(effective_threads(256, 4, 64), 4u);
}

TEST(EffectiveThreads, CappedAtFourTimesHardwareThreads) {
  EXPECT_EQ(effective_threads(1'000'000, 1'000'000, 1), kMaxThreadsPerCore * hw_threads());
}

TEST(EffectiveThreads, EveryThreadGetsAtLeastMinRows) {
  EXPECT_EQ(effective_threads(100, 8, 64), 1u);  // 100 rows / 64 = 1 thread
  EXPECT_EQ(effective_threads(200, 8, 64), 3u);  // 200 / 64 = 3
  EXPECT_EQ(effective_threads(3, 8, 1), 3u);     // never more threads than rows
}

TEST(EffectiveThreads, DegenerateInputsGiveOneThread) {
  EXPECT_EQ(effective_threads(0, 4, 64), 1u);
  EXPECT_EQ(effective_threads(0, 0, 0), 1u);
  EXPECT_EQ(effective_threads(10, 4, 0), 4u);  // min_rows_per_thread 0 is treated as 1
}

TEST(ParallelForRows, FourThreadsRunFourContiguousChunksOnFourOsThreads) {
  const std::size_t rows = 257;
  const auto chunks = record_chunks([&](auto fn) { parallel_for_rows(rows, 4, fn); });
  ASSERT_EQ(chunks.size(), 4u);
  std::set<std::thread::id> ids;
  std::size_t expected_begin = 0;
  for (const auto& c : chunks) {
    EXPECT_EQ(c.begin, expected_begin);
    // Near-equal split: 257 = 65 + 64 + 64 + 64.
    EXPECT_TRUE(c.end - c.begin == 64 || c.end - c.begin == 65) << c.begin << ".." << c.end;
    expected_begin = c.end;
    ids.insert(c.thread);
  }
  EXPECT_EQ(expected_begin, rows);
  EXPECT_EQ(ids.size(), 4u);
  EXPECT_EQ(chunks.front().thread, std::this_thread::get_id()) << "chunk 0 runs on the caller";
}

TEST(ParallelForRows, SingleThreadRunsWholeRangeOnCaller) {
  const auto chunks = record_chunks([](auto fn) { parallel_for_rows(100, 1, fn); });
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].begin, 0u);
  EXPECT_EQ(chunks[0].end, 100u);
  EXPECT_EQ(chunks[0].thread, std::this_thread::get_id());
}

// run_rows is exactly what gemv/gemm call. If effective_threads (or run_rows) collapsed every
// request to one thread, the bit-identical threading tests would still pass; this one would not.
TEST(RunRows, UsesTheRequestedNumberOfOsThreads) {
  const auto chunks = record_chunks([](auto fn) { run_rows(257, 4, 1, fn); });
  std::set<std::thread::id> ids;
  for (const auto& c : chunks) ids.insert(c.thread);
  EXPECT_EQ(chunks.size(), 4u);
  EXPECT_EQ(ids.size(), 4u);
}

TEST(RunRows, DefaultMinRowsKeepsSmallProblemsOnTheCaller) {
  const auto chunks = record_chunks([](auto fn) { run_rows(100, 4, 64, fn); });
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(chunks[0].thread, std::this_thread::get_id());
}

TEST(ParallelForRows, ExceptionInWorkerIsRethrownOnCaller) {
  std::mutex mutex;
  std::size_t rows_done = 0;
  EXPECT_THROW(parallel_for_rows(400, 4,
                                 [&](std::size_t begin, std::size_t end) {
                                   if (begin == 200) throw std::runtime_error("worker failed");
                                   const std::lock_guard<std::mutex> lock(mutex);
                                   rows_done += end - begin;
                                 }),
               std::runtime_error);
  EXPECT_EQ(rows_done, 300u) << "the other three chunks still ran and were joined";
}

TEST(ParallelForRows, ExceptionOnCallingThreadIsRethrownAfterJoin) {
  std::mutex mutex;
  std::size_t rows_done = 0;
  EXPECT_THROW(parallel_for_rows(400, 4,
                                 [&](std::size_t begin, std::size_t end) {
                                   if (begin == 0) throw std::logic_error("caller chunk failed");
                                   const std::lock_guard<std::mutex> lock(mutex);
                                   rows_done += end - begin;
                                 }),
               std::logic_error);
  EXPECT_EQ(rows_done, 300u);
}
