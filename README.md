# Adaptive WALNUTS in C++

This is a C++ implementation of the following three [Hamiltonian Monte
Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (HMC) samplers.

* [NUTS](https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)
* [WALNUTS](https://arxiv.org/abs/2506.18746)
* Adaptive WALNUTS (continuous form of [Nutpie](https://github.com/pymc-devs/nutpie)-style adaptation)


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

Running Stan models requires the BridgeStan interface.  See the BridgeStan documentation for more
information on its dependencies.

* [BridgeStan](https://github.com/roualdes/bridgestan)  ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause))

## Using WALNUTS in a C++ project

This library is header only and only requires Eigen (also header only)
to run (additional dependencies are required for testing and documtnation).
If your project uses CMake, you can depend on our
`walnuts` library target. If not, any method of adding the `include/`
folder of this repository to your build system's include paths should suffice
as long as you also provide Eigen yourself.

## Building the examples and tests

CMake is required to build the examples and tests.

### Configuring the build

The basic configuration is

```sh
cmake <options> <repo_root>
```

where `<options>` are the CMake options and `<repo_root>` is the root
directory of the repository (where `CMakeLists.txt` is found).

Some common options are:

- `-B <build_dir>` - Specify the build directory where the build files will be generated. If omitted, the directory you run the command from will be used.
- `-DCMAKE_BUILD_TYPE=Release` - Set the build type to Release.
- `-DWALNUTS_BUILD_TESTS=ON` - Enable building of the tests (currently on by default).
- `-DWALNUTS_BUILD_EXAMPLES=ON` - Enable building of the examples (currently on by default).
- `-DWALNUTS_BUILD_DOC=ON` - Enable building of the documentation (currently on by default).
- `-DWALNUTS_USE_MIMALLOC=ON` - Link against the [mimalloc](https://github.com/microsoft/mimalloc), a MIT licensed custom memory allocator which can improve performance.
- `-DWALNUTS_BUILD_STAN=ON` - Enable the example program which uses Stan via [BridgeStan](github.com/roualdes/bridgestan).

Other options can be found in the CMake help output or [documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html).

For example, a basic configuration which creates a `./build` directory in the repo
root can be done with

```sh
cmake . -B ./build -DCMAKE_BUILD_TYPE=Release
```

The remaining instructions assume that commands are run from whatever
directory you specified as the build directory (e.g., `./build` in the above command).

### Building

The easiest way to build the project is with the `cmake --build`
command. This will build all available executable targets by default.

For example, to build and run the example:

```bash
cmake --build . --target examples
./examples
```


### Testing

Running the tests is easiest with the `ctest` command distributed with CMake.

```bash
# assuming you did _not_ specify -DWALNUTS_BUILD_TESTS=OFF earlier...
cmake --build . --parallel 4
ctest
```

### Documentation

To build the C++ documentation using Doxygen:

```bash
cmake --build . --target doc
```

The root of the generated doc will be found in

* `./html/index.html`.


## Project overview

The project directory structure is as follows.


```
.
├── examples
│   └── .cpp files, one per example
├── include
│   └── walnuts
│       └── .hpp files containing the library source code
├── tests
│   ├── .cpp files, one per test
│   └── CMakeLists.txt
├── CMakeLists.txt
└── README.md
```
