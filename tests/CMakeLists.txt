if(MSVC)
  set(gtest_force_shared_crt on)
endif()

FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main         )
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

FetchContent_Declare(boost_ut
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/boost-ext/ut.git
        GIT_TAG v2.3.1)
FetchContent_MakeAvailable(boost_ut)


function(add_gtest_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)

  target_link_libraries(${TEST_NAME}
    PRIVATE
      gtest_main
      Eigen3::Eigen
      nuts::nuts
  )

  gtest_discover_tests(${TEST_NAME})
endfunction()

add_gtest_test(util_test)


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

add_library(boost_ut_runner OBJECT test_runner.cpp)
target_link_libraries(boost_ut_runner PUBLIC Boost::ut)

function(add_boost_ut_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
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

