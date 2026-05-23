if(MSVC)
  set(gtest_force_shared_crt on)
endif()

set(BUILD_GMOCK OFF CACHE BOOL "Disable building Google Mock" FORCE)
FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(add_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE})
  target_link_libraries(${TEST_NAME}
    PRIVATE
      gtest_main
      Eigen3::Eigen
      walnuts
  )
  gtest_discover_tests(${TEST_NAME})
endfunction()

add_test(summary_test summary_test.cpp)
