cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only NUTS sampler library"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()
##########################
## Cmake Options        ##
##########################
option(WALNUTS_BUILD_EXAMPLES "Build the example targets for the library" ON)
# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING
    "Choose the type of build, options are: None Debug Release"
    FORCE)

##########################
## Global Dependencies  ##
##########################
find_package(Git REQUIRED)

# where to put Eigen
set(EIGEN_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/eigen")
set(EIGEN_TAG 3.4.0)
# only clone on first configure
if(NOT EXISTS "${EIGEN_SRC_DIR}/Eigen")
  message(STATUS "Cloning Eigen ${EIGEN_TAG}…")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone --branch ${EIGEN_TAG} --depth 1 -- https://gitlab.com/libeigen/eigen.git ${EIGEN_SRC_DIR}
    RESULT_VARIABLE _git_result
  )
  if(NOT _git_result EQUAL 0)
    message(FATAL_ERROR "Git clone of Eigen failed")
  endif()
endif()

# now expose it as a pure‐header interface target
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE
  "${EIGEN_SRC_DIR}"
)

# where to put Eigen
set(EIGENRAND_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/eigenrand")
set(EIGENRAND_TAG v0.5.1)
# only clone on first configure
if(NOT EXISTS "${EIGENRAND_SRC_DIR}/EigenRand")
  message(STATUS "Cloning EigenRand ${EIGENRAND_TAG}…")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone --branch ${EIGENRAND_TAG} --depth 1 -- https://github.com/bab2min/EigenRand.git ${EIGENRAND_SRC_DIR}
    RESULT_VARIABLE _git_result
  )
  if(NOT _git_result EQUAL 0)
    message(FATAL_ERROR "Git clone of EigenRand failed")
  endif()
endif()

# now expose it as a pure‐header interface target
add_library(EigenRand::EigenRand INTERFACE IMPORTED)
target_include_directories(EigenRand::EigenRand INTERFACE
  "${EIGENRAND_SRC_DIR}"
)


#############################
## Making Walnuts Library  ##
#############################

# 1) Define your header-only interface target
add_library(walnuts INTERFACE)
target_include_directories(walnuts INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>                            # for installs
)
# 2) Create the “namespace” alias
add_library(walnuts::walnuts ALIAS walnuts)

##########################
##       Example        ##
##########################
if (WALNUTS_BUILD_EXAMPLES)
  add_executable(test_nuts ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
  target_link_libraries(test_nuts PRIVATE Eigen3::Eigen EigenRand::EigenRand walnuts::walnuts)
  target_compile_options(test_nuts PRIVATE -O3 -Wall)
endif()


##########################
##       Extras         ##
##########################
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extras/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extras")
endif()
