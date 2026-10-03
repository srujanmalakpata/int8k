cmake_minimum_required(VERSION 3.21)
project(int8k VERSION 0.1.0 DESCRIPTION "Quantized INT8/INT4 inference kernels" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(INT8K_BUILD_TESTS "Build the GoogleTest suite" ON)
option(INT8K_BUILD_BENCH "Build the benchmark binary" ON)
option(INT8K_BUILD_TOOLS "Build the int8k-cli tool" ON)
option(INT8K_WERROR "Treat compiler warnings as errors" ON)
option(INT8K_SANITIZE "Build everything with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(INT8K_TSAN "Build everything with ThreadSanitizer (data races in the row split)" OFF)
option(INT8K_PORTABLE "Compile out the x86 AVX2 kernels (tests the non-x86 fallback)" OFF)
option(INT8K_FETCH_GTEST "Download GoogleTest with FetchContent if find_package fails (CI only)" OFF)

find_package(Threads REQUIRED)

# ---- Warnings -----------------------------------------------------------------------------
# Baseline warnings for everything we compile; the library itself also gets conversion checks.
add_library(int8k_warnings INTERFACE)
target_compile_options(int8k_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
if(INT8K_WERROR)
  target_compile_options(int8k_warnings INTERFACE -Werror)
endif()

add_library(int8k_strict_warnings INTERFACE)
target_link_libraries(int8k_strict_warnings INTERFACE int8k_warnings)
target_compile_options(int8k_strict_warnings INTERFACE -Wconversion -Wsign-conversion)

# ---- Sanitizers ---------------------------------------------------------------------------
if(INT8K_SANITIZE AND INT8K_TSAN)
  message(FATAL_ERROR "INT8K_SANITIZE (ASan) and INT8K_TSAN cannot be combined")
endif()
if(INT8K_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(INT8K_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# ---- Library ------------------------------------------------------------------------------
add_library(int8k
  src/cpu.cpp
  src/gemv.cpp
  src/kernels_avx2.cpp
  src/kernels_scalar.cpp
  src/quantize.cpp
  src/random.cpp
  src/reference.cpp
  src/stats.cpp)
add_library(int8k::int8k ALIAS int8k)
target_include_directories(int8k
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(int8k PUBLIC cxx_std_20)
if(INT8K_PORTABLE)
  target_compile_definitions(int8k PRIVATE INT8K_FORCE_PORTABLE)
endif()
target_link_libraries(int8k PUBLIC Threads::Threads PRIVATE int8k_strict_warnings)

# ---- Tools / bench ------------------------------------------------------------------------
if(INT8K_BUILD_TOOLS)
  add_executable(int8k-cli tools/int8k_cli.cpp)
  target_link_libraries(int8k-cli PRIVATE int8k int8k_strict_warnings)
endif()

if(INT8K_BUILD_BENCH)
  add_executable(int8k-bench bench/bench_main.cpp)
  target_link_libraries(int8k-bench PRIVATE int8k int8k_strict_warnings)
endif()

# ---- Tests --------------------------------------------------------------------------------
if(INT8K_BUILD_TESTS)
  enable_testing()
  include(cmake/GTest.cmake)
  add_subdirectory(tests)
endif()
