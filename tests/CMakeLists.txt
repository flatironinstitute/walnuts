if(MSVC)
  set(gtest_force_shared_crt on)
endif()

FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Set up testing framework, then add tests
# define a function to add a gtest‐based nuts test
function(add_nuts_test TEST_NAME TEST_SOURCE)
  # create the executable
  add_executable(${TEST_NAME} ${TEST_SOURCE})

  # link in GoogleTest, Eigen, and your nuts lib
  target_link_libraries(${TEST_NAME}
    PRIVATE
      gtest_main
      Eigen3::Eigen
      nuts::nuts
  )

  #  register it with CTest
  gtest_discover_tests(${TEST_NAME})
endfunction()

add_nuts_test(combine_span_test combine_span_test.cpp)
add_nuts_test(online_moments_test online_moments_test.cpp)
add_nuts_test(dual_average_test dual_average_test.cpp)
add_nuts_test(util_test util_test.cpp)
