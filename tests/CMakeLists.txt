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

  add_test(
    NAME ${TEST_NAME}
    COMMAND $<TARGET_FILE:${TEST_NAME}>
  )

  target_sources(jumbo_test PRIVATE ${TEST_NAME}.cpp)
endfunction()

add_boost_ut_test(combine_span_test)
add_boost_ut_test(dual_average_test)
add_boost_ut_test(online_moments_test)
add_boost_ut_test(util_test)

