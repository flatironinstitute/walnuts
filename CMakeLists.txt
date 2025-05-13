cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only NUTS sampler library"
  LANGUAGES CXX
)

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)
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
include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_Populate(Eigen)   # ⇒ eigen_SOURCE_DIR / eigen_BINARY_DIR
# 1) Create an INTERFACE library for the headers
add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers
  INTERFACE
    ${eigen_SOURCE_DIR}      # root of Eigen repo: contains Eigen/ and unsupported/
)
# 2) Alias it so downstream CMakeLists can do find-like targets
add_library(Eigen3::Eigen ALIAS eigen_headers)


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
  target_link_libraries(test_nuts PRIVATE Eigen3::Eigen walnuts::walnuts)
  target_compile_options(test_nuts PRIVATE -O3 -Wall)
endif()


##########################
##       Extras         ##
##########################
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extras/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extras")
endif()
