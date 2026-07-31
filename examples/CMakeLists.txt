add_executable(examples examples.cpp)
target_link_libraries(examples PRIVATE walnutpie)
target_compile_options(examples PRIVATE -O3 -Wall)

option(WALNUTPIE_USE_TSAN "Build examples with ThreadSanitizer" OFF)

# Apply the standard example flags plus TSan if enabled
function(walnutpie_add_example name)
    add_executable (${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE walnutpie)
    target_compile_options(${name} PRIVATE -O3 -Wall)
    if(WALNUTPIE_USE_TSAN)
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

walnutpie_add_example(summary)

walnutpie_add_example(walnutpie_api)
if(UNIX AND NOT APPLE)
    target_link_libraries(walnutpie_api PRIVATE atomic)
endif()


FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.5.0
)
FetchContent_MakeAvailable(cli11)

add_executable(stan_cli stan_cli.cpp)
target_link_libraries(stan_cli PRIVATE
  walnutpie
  Eigen3::Eigen
  BridgeStan::BridgeStan
  CLI11::CLI11)
target_compile_options(stan_cli PRIVATE -Wall)
