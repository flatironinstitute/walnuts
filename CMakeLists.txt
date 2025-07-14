cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only NUTS sampler library"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()

##########################
## Cmake Options        ##
##########################
option(WALNUTS_USE_MIMALLOC "Statically link mimalloc" OFF)
option(WALNUTS_BUILD_EXAMPLES "Build the example targets for the library" ON)
option(WALNUTS_BUILD_STAN "Build Stan example with BridgeStan" OFF)
option(WALNUTS_BUILD_TESTS "Build the test targets for the library" ON)
# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING
    "Choose the type of build, options are: None Debug Release"
    FORCE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Enable -Weverything and other Clang-only warnings
  # After no-c++98 warnings, rest are to suppress gtest warnings
  add_compile_options(-Wall -Weverything -Wno-padded
  -Wno-poison-system-directories -Wno-ctad-maybe-unsupported
  -Wno-c++98-compat-pedantic -Wno-c++98-compat 
  -Wno-global-constructors -Wno-exit-time-destructors
  -Wno-covered-switch-default -Wno-switch-enum -Wno-weak-vtables
  -Wno-zero-as-null-pointer-constant -Wno-switch-default
  -Wno-missing-variable-declarations)

  # Optionally enable IWYU if found
  find_program(IWYU_PATH NAMES include-what-you-use iwyu)
  if (IWYU_PATH)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_PATH};-Xiwyu;--mapping_file=...")
  else()
    message(STATUS "include-what-you-use not found, skipping IWYU integration.")
  endif()
endif()


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

include(FetchContent)

set(WALNUTS_MIMALLOC_TAG "v2.2.3" CACHE STRING "Version of mimalloc to use")
if(WALNUTS_USE_MIMALLOC)
    option(MI_BUILD_TESTS OFF)
    option(MI_BUILD_SHARED OFF)
    option(MI_PADDING OFF)
    option(MI_DEBUG_FULL OFF)
    FetchContent_Declare(
            mimalloc
            GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
            GIT_TAG ${WALNUTS_MIMALLOC_TAG}
    )
    FetchContent_MakeAvailable(mimalloc)
endif()

set(WALNUTS_BRIDGESTAN_TAG "1d4ff6a10d572ee22ed26b0274059bb4fe885187" CACHE STRING "Version of BridgeStan to use")
if(WALNUTS_BUILD_STAN)
    FetchContent_Declare(
            bridgestan
            GIT_REPOSITORY https://github.com/roualdes/bridgestan.git
            GIT_TAG ${WALNUTS_BRIDGESTAN_TAG}
    )
    FetchContent_MakeAvailable(bridgestan)

    add_library(BridgeStan::BridgeStan INTERFACE IMPORTED)
    target_include_directories(BridgeStan::BridgeStan INTERFACE
      "${CMAKE_BINARY_DIR}/_deps/bridgestan-src/src"
    )

  add_executable(test_stan ${CMAKE_CURRENT_SOURCE_DIR}/examples/test_stan.cpp)
  target_link_libraries(test_stan PRIVATE Eigen3::Eigen nuts::nuts BridgeStan::BridgeStan)
  target_compile_options(test_stan PRIVATE -O3 -Wall)
endif()

#############################
## Making Walnuts Library  ##
#############################

# 1) Define your header-only interface target
add_library(walnuts INTERFACE)
target_include_directories(walnuts INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>                            # for installs
)

target_link_libraries(walnuts INTERFACE Eigen3::Eigen)

if (WALNUTS_USE_MIMALLOC)
    target_link_libraries(walnuts INTERFACE mimalloc-static)
endif()

# 2) Create the “namespace” alias
add_library(nuts::nuts ALIAS walnuts)

##########################
##       Example        ##
##########################
if (WALNUTS_BUILD_EXAMPLES)
  add_executable(test_nuts ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
  target_link_libraries(test_nuts PRIVATE nuts::nuts)
  target_compile_options(test_nuts PRIVATE -O3 -Wall)
endif()

##########################
##       Extras         ##
##########################
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extras/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extras")
endif()

##########################
##       Tests         ##
##########################
if (WALNUTS_BUILD_TESTS)
  enable_testing()
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

##########################
##       Format         ##
##########################
# Define list of files to format (you can make this recursive if desired)
file(GLOB_RECURSE ALL_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

find_program(CLANG_FORMAT_BIN clang-format)

if (CLANG_FORMAT_BIN)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
      -i
      -style=file
      ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
  )
else()
  message(WARNING "clang-format not found. 'format' target will not be available.")
endif()

##########################
##     Doxygen Docs     ##
##########################
#################################
##        Documentation        ##
#################################
find_package(Doxygen REQUIRED)

# Public headers only
file(GLOB_RECURSE WALNUT_HEADERS
     "${CMAKE_CURRENT_SOURCE_DIR}/include/walnuts/*.hpp")

# Doxygen configuration overrides
set(DOXYGEN_FILE_PATTERNS    "*.hpp")
set(DOXYGEN_GENERATE_HTML    YES)
set(DOXYGEN_GENERATE_LATEX   NO)
set(DOXYGEN_EXTRACT_ALL YES)

# `cmake --build . --target doc`
doxygen_add_docs(
  doc                         # target name
  ${WALNUT_HEADERS}           # INPUT files
  COMMENT "Generate API documentation in ${DOXYGEN_OUTPUT_DIRECTORY}"
)
