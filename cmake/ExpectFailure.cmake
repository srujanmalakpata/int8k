# CTest helper: runs a command and passes only if it exits with EXPECT_EXIT AND its stderr
# contains EXPECT_STDERR (a literal substring). WILL_FAIL alone would accept any non-zero exit,
# including a crash or an unrelated error, so it cannot tell WHICH rejection happened.
#
#   cmake -DEXPECT_EXIT=2 "-DEXPECT_STDERR=some text" -P ExpectFailure.cmake -- <command> [args...]
if(NOT DEFINED EXPECT_EXIT OR NOT DEFINED EXPECT_STDERR)
  message(FATAL_ERROR "ExpectFailure.cmake needs -DEXPECT_EXIT=... and -DEXPECT_STDERR=...")
endif()

set(command)
set(after_separator FALSE)
math(EXPR last_arg "${CMAKE_ARGC} - 1")
foreach(i RANGE ${last_arg})
  if(after_separator)
    list(APPEND command "${CMAKE_ARGV${i}}")
  elseif(CMAKE_ARGV${i} STREQUAL "--")
    set(after_separator TRUE)
  endif()
endforeach()
if(NOT command)
  message(FATAL_ERROR "ExpectFailure.cmake: no command after '--'")
endif()

execute_process(COMMAND ${command} RESULT_VARIABLE exit_code OUTPUT_VARIABLE out
                ERROR_VARIABLE err)
string(FIND "${err}" "${EXPECT_STDERR}" found)
if(NOT exit_code STREQUAL EXPECT_EXIT)
  message(FATAL_ERROR "expected exit code ${EXPECT_EXIT}, got '${exit_code}'\nstderr:\n${err}")
endif()
if(found EQUAL -1)
  message(FATAL_ERROR "stderr does not contain '${EXPECT_STDERR}'\nstderr:\n${err}")
endif()
message(STATUS "exit ${exit_code} with expected message: ${EXPECT_STDERR}")
