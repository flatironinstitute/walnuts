

FetchContent_Declare(googletest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

# Let's cmake know we are going to use ctest and to attach this test to it.
enable_testing()
add_executable(mock_test mock_test.cpp)
target_link_libraries(mock_test
  gtest_main
  Eigen3::Eigen
  nuts::nuts)
add_test(NAME mock_test COMMAND mock_test)
