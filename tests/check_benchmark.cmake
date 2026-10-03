# The displayed best float32 GEMV latency must also be the speedup denominator.
string(REPLACE "|" ";" EMULATOR "${EMULATOR}")
execute_process(COMMAND ${EMULATOR} "${BENCH}" --quick --reps 3 --threads 1
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Benchmark failed (${result}): ${error}")
endif()

set(backend scalar)
if(output MATCHES "avx2=yes")
  set(backend avx2)
endif()
if(NOT output MATCHES "\\| f32 gemv +\\| ${backend} +\\| +1 +\\| +1 +\\|[^\n]*\\| +1\\.00x +\\|")
  message(FATAL_ERROR "Best float32 GEMV row must report 1.00x against itself:\n${output}")
endif()
