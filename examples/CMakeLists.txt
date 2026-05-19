# Configure with cmake -DWALNUTS_USE_TSAN=ON .. to enable TSan

option(WALNUTS_USE_TSAN "Build examples with ThreadSanitizer" OFF)

if(WALNUTS_USE_TSAN)
  message(STATUS "TSan enabled — suppressions file: ${CMAKE_SOURCE_DIR}/tsan_suppressions.txt")
endif() 

add_executable(summary summary.cpp)
target_link_libraries(summary PRIVATE walnuts)
target_compile_options(summary PRIVATE -O3 -Wall)

add_executable(walnuts_api walnuts_api.cpp)
target_link_libraries(walnuts_api PRIVATE Eigen3::Eigen walnuts)
if (UNIX AND NOT APPLE)
  target_link_libraries(walnuts_api PRIVATE atomic)
endif()
target_compile_options(walnuts_api PRIVATE -O3 -Wall)

if(WALNUTS_USE_TSAN)
  target_compile_options(summary PRIVATE -fsanitize=thread)
  target_link_options(summary PRIVATE -fsanitize=thread)
  target_compile_options(walnuts_api PRIVATE -fsanitize=thread)
  target_link_options(walnuts_api PRIVATE -fsanitize=thread)

  add_custom_target(tsan_summary
    COMMAND ${CMAKE_COMMAND} -E env
      "TSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/tsan_suppressions.txt"
      $<TARGET_FILE:summary>
    DEPENDS summary
    COMMENT "Running summary with TSan suppressions"
  )
  add_custom_target(tsan_walnuts_api
    COMMAND ${CMAKE_COMMAND} -E env
      "TSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/tsan_suppressions.txt"
      $<TARGET_FILE:walnuts_api>
    DEPENDS walnuts_api
    COMMENT "Running walnuts_api with TSan suppressions"
  )
endif()

if(WALNUTS_BUILD_STAN)
  FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.5.0
  )
  FetchContent_MakeAvailable(cli11)
  add_executable(stan_cli stan_cli.cpp)
  target_link_libraries(stan_cli PRIVATE Eigen3::Eigen walnuts BridgeStan::BridgeStan CLI11::CLI11)
  target_compile_options(stan_cli PRIVATE -Wall)
endif()
