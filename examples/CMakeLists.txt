add_executable(examples examples.cpp)
target_link_libraries(examples PRIVATE nuts::nuts)
target_compile_options(examples PRIVATE -O3 -Wall)



if(WALNUTS_BUILD_STAN)
  add_executable(examples_stan examples_stan.cpp)
  target_link_libraries(examples_stan PRIVATE Eigen3::Eigen nuts::nuts BridgeStan::BridgeStan)
  target_compile_options(examples_stan PRIVATE -Wall)
endif()
