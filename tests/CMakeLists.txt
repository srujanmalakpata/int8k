add_executable(int8k_tests
  test_cpu.cpp
  test_quantize.cpp
  test_gemv.cpp
  test_gemm.cpp
  test_kernels.cpp
  test_parallel.cpp
  test_properties.cpp
  test_random.cpp)
# test_kernels.cpp and test_parallel.cpp include the internal headers in src/ to call the
# kernels and the row splitter directly, so they cannot be bypassed by the dispatcher.
target_include_directories(int8k_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                               ${PROJECT_SOURCE_DIR}/src)
if(INT8K_PORTABLE)
  target_compile_definitions(int8k_tests PRIVATE INT8K_FORCE_PORTABLE)
endif()
target_link_libraries(int8k_tests PRIVATE int8k int8k_warnings GTest::gtest_main)
gtest_discover_tests(int8k_tests DISCOVERY_TIMEOUT 60)

# End-to-end tests of the CLI binary: real process, real argument parsing, real exit codes.
if(INT8K_BUILD_TOOLS)
  add_test(NAME cli.info COMMAND int8k-cli info)
  set_tests_properties(cli.info PROPERTIES PASS_REGULAR_EXPRESSION "avx2 available: (yes|no)")

  add_test(NAME cli.quantize_q4 COMMAND int8k-cli quantize --rows 64 --cols 100 --format q4)
  set_tests_properties(cli.quantize_q4 PROPERTIES
    PASS_REGULAR_EXPRESSION "format: q4.*max_abs_error: [0-9.e+-]+.*snr_db: [0-9.]+")

  add_test(NAME cli.quantize_int8 COMMAND int8k-cli quantize --rows 17 --cols 33 --format int8)
  set_tests_properties(cli.quantize_int8 PROPERTIES PASS_REGULAR_EXPRESSION "format: int8")

  add_test(NAME cli.selfcheck COMMAND int8k-cli selfcheck --rows 129 --cols 300 --threads 3)
  set_tests_properties(cli.selfcheck PROPERTIES PASS_REGULAR_EXPRESSION "selfcheck: PASS")

  add_test(NAME cli.gemv COMMAND int8k-cli gemv --rows 64 --cols 300 --format q4 --threads 2)
  set_tests_properties(cli.gemv PROPERTIES
    PASS_REGULAR_EXPRESSION "weight_quantize_ms.*act_quantize\\+gemv_ms.*snr_db: [0-9.]+")

  # Tiny shapes have poor SNR (few values), but the exact-oracle check must still pass.
  add_test(NAME cli.selfcheck_tiny COMMAND int8k-cli selfcheck --rows 1 --cols 100 --threads 2)
  set_tests_properties(cli.selfcheck_tiny PROPERTIES PASS_REGULAR_EXPRESSION "selfcheck: PASS")

  # Each rejection test checks the exit code (2) AND the specific error message, so a crash or
  # an unrelated error cannot pass for the expected rejection.
  # The command runs inside a host `cmake -P`, so a cross build's emulator (qemu) is prepended
  # explicitly; CTest only adds it automatically when the test COMMAND is the target itself.
  function(int8k_expect_rejection name message)
    set(command ${ARGN})
    if(CMAKE_CROSSCOMPILING_EMULATOR)
      list(PREPEND command ${CMAKE_CROSSCOMPILING_EMULATOR})
    endif()
    add_test(NAME ${name}
             COMMAND ${CMAKE_COMMAND} -DEXPECT_EXIT=2 "-DEXPECT_STDERR=${message}"
                     -P ${PROJECT_SOURCE_DIR}/cmake/ExpectFailure.cmake -- ${command})
  endfunction()

  int8k_expect_rejection(cli.rejects_negative_threads
    "--threads must be an integer in [0, 1024], got '-1'"
    $<TARGET_FILE:int8k-cli> gemv --rows 8 --cols 8 --threads -1)
  int8k_expect_rejection(cli.rejects_zero_rows
    "--rows must be an integer in [1, 1048576], got '0'"
    $<TARGET_FILE:int8k-cli> quantize --rows 0 --cols 8)
  int8k_expect_rejection(cli.rejects_trailing_junk
    "--stddev must be a finite number > 0, got '1x'"
    $<TARGET_FILE:int8k-cli> quantize --rows 8 --cols 8 --stddev 1x)
  int8k_expect_rejection(cli.rejects_bad_format
    "--format must be int8 or q4, got 'int3'"
    $<TARGET_FILE:int8k-cli> quantize --format int3)
  int8k_expect_rejection(cli.rejects_bad_backend
    "unknown backend 'neon'"
    $<TARGET_FILE:int8k-cli> gemv --rows 8 --cols 8 --backend neon)
  int8k_expect_rejection(cli.rejects_unknown_option
    "unknown option --colz"
    $<TARGET_FILE:int8k-cli> quantize --colz 8)
  int8k_expect_rejection(cli.rejects_unknown_command
    "unknown command 'frobnicate'"
    $<TARGET_FILE:int8k-cli> frobnicate)
endif()

if(INT8K_BUILD_BENCH AND INT8K_BUILD_TOOLS)
  int8k_expect_rejection(bench.rejects_bad_threads
    "--threads must be an integer in [0, 1024], got 'abc'"
    $<TARGET_FILE:int8k-bench> --threads abc)
  int8k_expect_rejection(bench.rejects_bad_reps
    "--reps must be an integer in [1, 100000], got '0'"
    $<TARGET_FILE:int8k-bench> --reps 0)
endif()
