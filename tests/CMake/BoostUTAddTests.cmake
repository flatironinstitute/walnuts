# based on https://gitlab.kitware.com/cmake/cmake/-/blob/master/Modules/GoogleTestAddTests.cmake?ref_type=heads
# this file defines a script that lists all the tests in a given executable
# and adds each one individually as a ctest test

set(script)
set(suite)
set(tests)

function(add_command NAME)
  set(_args "")
  foreach(_arg ${ARGN})
    if(_arg MATCHES "[^-./:a-zA-Z0-9_]")
      set(_args "${_args} [==[${_arg}]==]")
    else()
      set(_args "${_args} ${_arg}")
    endif()
  endforeach()
  set(script "${script}${NAME}(${_args})\n" PARENT_SCOPE)
endfunction()

# Run test executable to get list of available tests
if(NOT EXISTS "${TEST_EXECUTABLE}")
  message(FATAL_ERROR
    "Specified test executable '${TEST_EXECUTABLE}' does not exist"
  )
endif()
execute_process(
  COMMAND "${TEST_EXECUTABLE}" --list
  OUTPUT_VARIABLE output
  RESULT_VARIABLE result
)
if(NOT ${result} EQUAL 0)
  message(FATAL_ERROR
    "Error running test executable '${TEST_EXECUTABLE}':\n"
    "  Result: ${result}\n"
    "  Output: ${output}\n"
  )
endif()

string(REPLACE "\n" ";" output "${output}")

foreach(line ${output})
    # strip whitespace
    string(REGEX REPLACE " +" "" test "${line}")
    add_command(add_test
                "${test}"
                "${TEST_EXECUTABLE}"
                "${test}"
    )
    message(CONFIGURE_LOG "Discovered test: ${test}")
    list(APPEND tests "${test}")

endforeach()

# Create a list of all discovered tests, which users may use to e.g. set
# properties on the tests
add_command(set ${TEST_LIST} ${tests})

# Write CTest script
file(WRITE "${CTEST_FILE}" "${script}")
