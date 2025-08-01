FetchContent_Declare(boost_ut
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG v2.3.1)
FetchContent_MakeAvailable(boost_ut)



# this target builds an executable that contains all the tests
# this can be faster than building each separately for local development
add_executable(jumbo_test EXCLUDE_FROM_ALL test_runner.cpp)
set_target_properties(jumbo_test PROPERTIES
  CXX_STANDARD 20
  UNITY_BUILD ON
 )
target_link_libraries(jumbo_test
  PRIVATE
    nuts::nuts
    Eigen3::Eigen
    Boost::ut
)

# We build a 'main' object that is linked into each test
add_library(boost_ut_runner OBJECT test_runner.cpp)
target_link_libraries(boost_ut_runner PUBLIC Boost::ut)

function(add_boost_ut_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  # boost.UT requires C++20, but our project overall is only C++17
  set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 20)

  target_link_libraries(${TEST_NAME}
    PRIVATE
      boost_ut_runner
      Eigen3::Eigen
      nuts::nuts
  )

  # alternative to add_test() that discovers each test in the executable
  discover_boost_ut_test(${TEST_NAME})

  target_sources(jumbo_test PRIVATE ${TEST_NAME}.cpp)
endfunction()


# Based on https://gitlab.kitware.com/cmake/cmake/-/blob/master/Modules/GoogleTest.cmake#L545
# This defines a function that is an alternative to add_test()
# that adds each individual test in the executable as its own ctest test
function (discover_boost_ut_test TARGET)

  set(ctest_file_base "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}")
  set(ctest_include_file "${ctest_file_base}_include.cmake")
  set(ctest_tests_file "${ctest_file_base}_tests.cmake")

  file(WRITE "${ctest_include_file}"
      "if(EXISTS \"${ctest_tests_file}\")\n"
      "  include(\"${ctest_tests_file}\")\n"
      "else()\n"
      "  add_test(${TARGET}_NOT_BUILT ${TARGET}_NOT_BUILT)\n"
      "endif()\n"
  )

  add_custom_command(
    TARGET ${TARGET} POST_BUILD
    BYPRODUCTS "${ctest_tests_file}"
    COMMAND "${CMAKE_COMMAND}"
            -D "TEST_TARGET=${TARGET}"
            -D "TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>"
            -D "CTEST_FILE=${ctest_tests_file}"
            -P "${_UT_DISCOVER_TESTS_SCRIPT}"
    VERBATIM
  )


  # Add discovered tests to directory TEST_INCLUDE_FILES
  set_property(DIRECTORY
    APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_include_file}"
  )

endfunction()


set(_UT_DISCOVER_TESTS_SCRIPT
  ${CMAKE_CURRENT_LIST_DIR}/BoostUTAddTests.cmake
)
