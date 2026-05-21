option(WALNUTS_USE_TSAN "Build examples with ThreadSanitizer" OFF)

# Apply the standard example flags plus TSan if enabled
function(walnuts_add_example name)
  target_compile_options(${name} PRIVATE -O3 -Wall)
  if(WALNUTS_USE_TSAN)
    target_compile_options(${name} PRIVATE -fsanitize=thread)
    target_link_options(${name} PRIVATE -fsanitize=thread)
    add_custom_target(tsan_${name}
      COMMAND ${CMAKE_COMMAND} -E env
        $<TARGET_FILE:${name}>
      DEPENDS ${name}
      COMMENT "Running ${name} with TSan suppressions"
    )
  endif()
endfunction()

add_executable(summary summary.cpp)
target_link_libraries(summary PRIVATE walnuts)
walnuts_add_example(summary)

add_executable(walnuts_api walnuts_api.cpp)
target_link_libraries(walnuts_api PRIVATE walnuts)
if(UNIX AND NOT APPLE)
  target_link_libraries(walnuts_api PRIVATE atomic)
endif()
walnuts_add_example(walnuts_api)

if(WALNUTS_BUILD_STAN)
  FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.5.0
  )
  FetchContent_MakeAvailable(cli11)

  add_executable(stan_cli stan_cli.cpp)
  target_link_libraries(stan_cli PRIVATE walnuts Eigen3::Eigen BridgeStan::BridgeStan CLI11::CLI11)
  target_compile_options(stan_cli PRIVATE -Wall)
endif()
