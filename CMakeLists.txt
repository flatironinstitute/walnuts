cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only Walnut Pie sampler library"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)


##########################
## Top-Level Options   ##
##########################
option(WALNUTS_BUILD_EXAMPLES "Build the example targets for the library" ON)
option(WALNUTS_BUILD_STAN "Build Stan example with BridgeStan" OFF)
option(WALNUTS_BUILD_TESTS "Build the test targets for the library" OFF)
option(WALNUTS_BUILD_DOCS "Build the documentation for the library" OFF)
option(WALNUTS_USE_MIMALLOC "Statically link mimalloc" OFF) 
option(WALNUTS_COVERAGE "Enable source-based test coverage instrumentation" OFF)

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
# this is working around an issue with Eigen that may no longer be an issue  
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
    set(MI_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_PADDING      OFF CACHE BOOL "" FORCE)
    set(MI_DEBUG_FULL   OFF CACHE BOOL "" FORCE)
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
endif()

#############################
## Windows pathces         ##
#############################

if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_link_options(-static-libgcc -static-libstdc++ -static -lpthread)
endif()

#############################
## Making Walnuts Library  ##
#############################

add_library(walnuts INTERFACE)
target_include_directories(walnuts INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>                            # for installs
)

target_link_libraries(walnuts INTERFACE Eigen3::Eigen)

if (WALNUTS_USE_MIMALLOC)
    target_link_libraries(walnuts INTERFACE mimalloc-static)
endif()


##########################
##       Examples       ##
##########################
if (WALNUTS_BUILD_EXAMPLES)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/examples")
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
  "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp"
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
if (WALNUTS_BUILD_DOCS)
  find_package(Doxygen REQUIRED)

  file(GLOB_RECURSE WALNUT_HEADERS
      "${CMAKE_CURRENT_SOURCE_DIR}/include/walnuts/*.hpp")

  set(DOXYGEN_FILE_PATTERNS    "*.hpp")
  set(DOXYGEN_GENERATE_HTML    YES)
  set(DOXYGEN_GENERATE_LATEX   NO)
  set(DOXYGEN_EXTRACT_ALL      NO)
  set(DOXYGEN_CITE_BIB_FILES   "${CMAKE_CURRENT_SOURCE_DIR}/docs/refs.bib")

  doxygen_add_docs(
    doc
    ${WALNUT_HEADERS}
    COMMENT "Generate API documentation in ${DOXYGEN_OUTPUT_DIRECTORY}"
  )
endif()


