# WALNUTS in C++

First, we are building and integrating NUTS.  Second, we will extend to WALNUTS.

## Compiling and running

With your own version of Eigen, you can ignore the cmake and call the compiler as normal.

```bash
cd walnuts_cpp
clang++ -std=c++17 -O3 -I lib/eigen-3.4.0 -I ./include ./examples/test.cpp -o test
./test
```

Alternatively you can use the cmake setup.

```bash
# /usr/bin/bash
# Assumes you start in walnuts_cpp directory

# Run cmake with
# The source directory as our current directory (-S .)
# and build in a new folder "build" (-B "build")
# Note that we use cmake's fetch_content and ExternalProject_Add
#  which manage our dependencies for us (just Eigen by default)
#  This means that creating the project from source
#  does require an internet connection.
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE

# After this call all of our build dependencies
# and make targets now exist in build
ls ./build
cd build

# List possible targets
make help

# Build and run the test_nuts example in ./example/
# Note: cmake will pull Eigen 3.4 down from gitlab
#  It will only do this the first time you run
#  a make command that depends on Eigen.
make -j3 test_nuts
./test_nuts
```

## Structure

```bash
walnuts_cpp
├── examples # Test Examples
├── extras # Folder for optional dev tools
├── include # Headers for the library
│   └── walnuts
```

## CMake Tips

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

### Refresh CMake

Cmake stores a `CMakeCache.txt` file with the variables from your most recent build.
For an existing build you want to completely refresh use `--fresh` when building.

```bash
# /usr/bin/bash
# From walnuts_cpp
# Run a build but use --fresh to force
# a hard reset of all cached variables
cmake -S . -B "build" --fresh

# All the cmake targets now exist in build
cd build
```

### View Project Targets

To see the available targets from the top level directory run the following after building

```bash
# /usr/bin/bash
# From walnuts_cpp
cmake -S . -B "build"
cmake --build build --target help
```

We can also use this to build from the top level directory

```bash
# /usr/bin/bash
# From walnuts_cpp
# This will take longer as we include depedencies
# google test and google benchmark
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE

# Now the build directory is setup and we we can build and run the benchmarks
cmake --build build --parallel 3 --target test_nuts
```

When in the `build` directory you can call `cmake ..` to run cmake again.
This is nice for refreshing variables with `cmake .. --fresh`

### Debugging Options With CMake

Setting `-DCMAKE_BUILD_TYPE=DEBUG` will make the make file generation verbose.
For all other build types you can add `VERBOSE=1` to your make call to see a trace of the actions CMake performs.

## Formatting

```bash
pushd walnuts_cpp/include/walnuts
clang-format -i -style=Google *.hpp
popd

cd walnuts_cpp/examples/
clang-format -i -style=Google *.hppcp 
```

We chose Google largely style because it's compact.

