add_executable(walnuts_api walnuts_api.cpp) 
target_link_libraries(walnuts_api PRIVATE Eigen3::Eigen walnuts)
target_compile_options(walnuts_api PRIVATE -O3 -Wall) 
target_compile_options(walnuts_api PRIVATE -fsanitize=thread) 
target_link_options(walnuts_api PRIVATE -fsanitize=thread) 

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
