# Adaptive WALNUTS in C++

This is a C++ implementation of the following three Hamiltonian Monte
Carlo (HMC) samplers.

* [NUTS](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
* [WALNUTS](https://arxiv.org/abs/2506.18746)
* Adaptive WALNUTS


## Licensing

The project is distributed under the following licenses.

* Code: [MIT License](https://opensource.org/license/mit)
* Documentation: [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)


## Dependencies

The dependencies may all be downloaded through CMake (see the next
section).

### Required build dependencies

* [Eigen C++ template library for linear algebra](https://eigen.tuxfamily.org/index.php?title=Main_Page)
([MPLv2 licensed](https://www.mozilla.org/en-US/MPL/2.0/))

### Required test dependencies

* [Google test](https://github.com/google/googletest) ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause))

### Required documentation dependencies

* [Doxygen](https://www.doxygen.nl/#google_vignette) ([GPLv1 licensed](https://www.gnu.org/licenses/old-licenses/gpl-1.0.html))

### Optional build dependences

For running Stan models, the following Stan interface that also
includes Stan is required.  See the BridgeStan documentation for more
information on its dependencies.

* [BridgeStan](https://github.com/roualdes/bridgestan)  ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause))


## CMake Preliminary Build

All build steps use CMake. The root configuration file is:

* `CMakeLists.txt`

All instructions assume a `bash` shell (typically at `/usr/bin/bash`)
and that commands are run from the top-level `walnuts` directory of
the repository.

### Internet Connection Required

The build uses CMake's `FetchContent` and `ExternalProject_Add` to
manage dependencies. **An internet connection is required when first
building the project.**

### Initial Configuration

**Before building any targets**, you must configure the build with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

The options used are as follows.

* `-S .`: Source is the current directory.
* `-B build`: Build directory will be `./build`.
* `-DCMAKE_BUILD_TYPE=Release`: Use Debug for debugging builds.

### Inspect build targets

To see available build targets:

```bash
cmake --build build --target help
```


## Building Targets

Once the [Preliminary build](#preliminary-build)above) has been run,
you can build individual targets from the top-level directory.

### Run example code

The file `examples/test.cpp` contains examples using NUTS, WALNUTS,
and adaptive WALNUTS. The first build will trigger a download of the
Eigen library.

To build and run the examples:

```bash
cmake --build build --parallel 3 --target test_nuts
./build/test_nuts
```


### Run unit tests

To build and run the unit tests:

```bash
cmake --build build --parallel 3 --target online_moments_test dual_average_test util_test
ctest --test-dir ./build/tests 
```

### Build documentation

To build the C++ documentatino using Doxygen:

```bash
cmake --build build --target doc  
```

The root of the generated doc will be found in

* `build/html/index.html`.


### Format code

Automatic code formatting applies to files

* `.hpp` files in `include`,
* `.cpp` files in `examples`, and
* `.cpp` files in `tests`. 

To automatically format C++ code using Clang Format:

```bash
cmake --build build --target format
```

#### Formatting style

The formatting style is defined in top-level file

* `.clang_format`.

The style is based on the Google style guide, with 

* braces/newline around all conditionals and loops, and
* sorted `#include` blocks.


## CMake Tips

### Refresh CMake

Cmake stores a `CMakeCache.txt` file with the variables from your most
recent build.  For an existing build you want to completely refresh
use `--fresh` when building.

```bash
# /usr/bin/bash

# remove old build, rebuild with --fresh to force a hard reset of cached variables
rm -rf ./build
cmake -S . -B "build" --fresh

# All the cmake targets now exist in build
cd build
```

### View Optional Project Flags

To view the optional flags for cmake with this project call `cmake -S . -B "build" -LH` and grep for cmake variables that start with walnuts.

```bash
# /usr/bin/bash
# Same as other command but -LH lists all cached cmake variables
# along with their help comment
cmake -S . -B "build" -LH | grep "WALNUTS" -B1

# Output
$ // Build the example targets for the library
$ WALNUTS_BUILD_EXAMPLES:BOOL=ON
```

### Include Variables When Compiling

To set variables when compiling cmake we use `-DVARIABLE_NAME=VALUE` like setting a macro.


### View Project Targets

To see the available targets from the top level directory run the following after building

```bash
# /usr/bin/bash
cmake -S . -B "build"
cmake --build build --target help
```

### Building from the top-level directory

We can also use this to build from the top level directory

```bash
# /usr/bin/bash

# This will take longer as we include dependencies
# google test and google benchmark
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE

# Now the build directory is setup and we we can build and run the benchmarks
cmake --build build --parallel 3 --target test_nuts
```

When in the `build` directory you can call `cmake ..` to run cmake again.
This is nice for refreshing variables with `cmake .. --fresh`

### Debugging Options With CMake

Setting `-DCMAKE_BUILD_TYPE=DEBUG` will make the make file generation
verbose.  For all other build types you can add `VERBOSE=1` to your
make call to see a trace of the actions CMake performs.


## Project directory structure

The project directory structure is as follows.  The `...` indicate
elided subdirectories.  The `build` directories are generated
automatically and the `lib` directory contains external includes, only
the top-level names of which are listed.


```bash
.
├── build...
├── examples
│   ├── test_stan.cpp
│   └── test.cpp
├── include
│   └── walnuts
│       ├── adaptive_walnuts.hpp 
│       ├── dual_average.hpp 
│       ├── nuts.hpp
│       ├── online_moments.hpp
│       ├── util.hpp 
│       └── walnuits.hpp
├── tests
│   ├── CMakeLists.txt 
│   ├── dual_average_test.cpp
│   ├── online_moments_test.cpp
│   └── util_test.cpp 
├── CMakeLists.txt
└── README.md
```
