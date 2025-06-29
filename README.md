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

The dependencies may all be downloaded through CMake (see the next section).

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


## Compiling and running with CMake

All of the build files use CMake.  The root CMake configuration is in

* Base CMake configuration: `walnuts_cpp/CMakelists.txt` 

All instructions below are coded for the `bash` shell, which is
conventionally located in in `/usr/bin/bash`.

All instructions assume the user **starts in the top-level directory**.

### Internet connection required

The build uses CMake's `fetch_content` and `ExternalProject_Add`
commands to manage dependencies.  This means that **creating 
the project from source requires an internet connection.**

### Base directory

All instructions assume the scripts start int he top level directory
`walnuts_cpp`.

### Preliminary build 

**Before anything else,** the following core build must be done to
prime CMake.  The script is **run from the top-level directory**.


```bash
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE
```

The cmake options are:

* `-S .` indicates that the source is in the current directory
* `-B "build"` indicates the build will happen in a a new folder
`build` under the top-level directory
* `-DCMAKE_BUILD_TYPE=RELEASE` indicates a release build (use `DEBUG`
for a debug build)

### Inspect build targets

After the call, the build dependences now exist in the `build`
directory. To see the make targets that may be built, run the
following.

```bash
cd build
make help
```

### Running example code

An example of NUTS, WALNUTS and adaptive WALNUTS may be found in.

* `examples/test.cpp`

Making it the first time will cause make to download the Eigen library.

```bash
cd build
make -j3 test_nuts
./test_nuts
```


## Running unit tests

The unit tests may be run with the following pair of commands to build
and run. Googletest must be installed (see dependencies above).

```bash 
cmake --build build --parallel 3 --target online_moments_test dual_average_test 
ctest --test-dir ./build/tests 
```


## Making documentation with Doxygen

The following commands may be used to make the C++ documentation.

```bash 
cd build
make doc 
```

The root of the generated doc will be found in
`walnuts_cpp/doc/html/index.html`.


## Automatic code formatting with clang-format

The following commands will use `clang-format` to automatically format
the following files.

* `.hpp` files in `include`
* `.cpp` files in `examples`
* `.cpp` files in `tests`.

```bash
cd build
make format
```

The formatting style is defined in the file
`walnuts_cpp/.clang_format`.  It specifies

* baseline Google format for compactness,
* braces and newlines around conditional and loop bodies, and
* includes sorted within block with blocks maintained.


## CMake Tips

### Refresh CMake 

Cmake stores a `CMakeCache.txt` file with the variables from your most recent build. 
For an existing build you want to completely refresh use `--fresh` when building. 

```bash 
# /usr/bin/bash 
# From walnuts_cpp 

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
# From walnuts_cpp
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
# From walnuts_cpp
cmake -S . -B "build"
cmake --build build --target help
```

### Building from the top-level directory

We can also use this to build from the top level directory

```bash
# /usr/bin/bash
# From walnuts_cpp

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
├── extras
│   └── readme.md
├── include
│   └── walnuts
│       ├── nuts.hpp
│       ├── util.hpp
│       └── walnuts.hpp
├── lib
│   ├── eigen-3.4.0...
├── tests
│   ├── CMakeLists.txt
│   └── mock_test.cpp
│   └── welford_test.cpp
├── CMakeLists.txt
└── README.md
```
