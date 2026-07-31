# Contributing to `walnutpie`

Thank you for your interest in contributing to `walnutpie`.
Contributions are managed through Github
[issues](https://github.com/flatironinstitute/walnuts/issues) and [pull
requests](https://github.com/flatironinstitute/walnuts/pulls).

We recommend opening a new issue or commenting on an existing issue before
beginning to discuss your planned contribution.


## Project overview

The project directory structure is as follows.

```
walnutpie/
├── CMakeLists.txt
├── pyproject.toml
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── include/                 # C++ library code
│   ├── walnutpie.hpp
│   └── walnutpie/
│       └── *.hpp
├── python/                  # Python interface
│   ├── CMakeLists.txt
│   ├── tests/
│   |   └── test_*.py
│   └── src/
│       └── walnutpie/
│           ├── *.py
│           ├── walnutpy.cpp
│           └── *.hpp
├── docs/                    # Documentation website source
│   └── *.rst
├── thirdparty/              # Vendored headers
├── examples/                # Example programs
│   ├── CMakeLists.txt
│   ├── *.hpp
│   └── *.cpp
└── tests/                   # C++ tests
    ├── CMakeLists.txt
    └── *_test.cpp

```


## Building the C++ library

We use [CMake](https://cmake.org/) to manage dependencies and build
our C++ code.


The basic configuration is to run the following command from the
top-level `walnutpie` directory.

```sh
cmake <options> <repo_root>
```

where `<options>` are the CMake options and `<repo_root>` is the root
directory of the repository (where `CMakeLists.txt` is found).

Some common options are:

- `-B <build_dir>` - Specify the build directory where the build files will be generated. If omitted, the directory you run the command from will be used.
- `-DCMAKE_BUILD_TYPE=Debug` - Set the build type to Debug.
- `-DCMAKE_BUILD_TYPE=Release` - Set the build type to Release.
- `-DWALNUTPIE_BUILD_TESTS=ON` - Enable building of the tests (currently on by default).
- `-DWALNUTPIE_BUILD_EXAMPLES=ON` - Enable building of the examples (currently on by default).
- `-DWALNUTPIE_USE_MIMALLOC=ON` - Link against the [mimalloc](https://github.com/microsoft/mimalloc), a MIT licensed custom memory allocator which can improve performance.
- `-DWALNUTPIE_USE_TSAN=ON` - Turn on the [thread sanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html)---only available if building with Clang.

Other options can be found in the CMake help output or [documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html).

For example, a basic configuration which creates a `./build` directory in the repo
root can be done with

```sh
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=RelWithDebug
```


Once configured, you can actually run a build with `cmake --build`:

```sh
cmake --build build/ -j4
```

### Running the C++ tests

We recommend using `ctest`, which is distributed with CMake.

```sh
cmake --build build/ -j4
cd build/
ctest --output-on-failure
```

#### Test coverage reports

Test coverage can be generated if you are using a LLVM-based toolchain.
To test code coverage during testing, you will have to specify the
top-level `cmake` call to include `DWALNUTPIE_COVERAGE=ON`.

The steps are to first run the test, directing the summary to the named
`.profraw` file.


```bash
LLVM_PROFILE_FILE="summary_test.profraw" ./tests/summary_test
```

Then, (using `xcrun` on a Mac), call `llvm-profdata` to merge the data into a
`.profdata` file.

```bash
xcrun llvm-profdata merge -sparse summary_test.profraw -o summary_test.profdata
```

Next, (also using `xcrun`), convert the generated `.profdata` file into html.

```bash
xcrun llvm-cov show ./tests/summary_test \
    -instr-profile=summary_test.profdata \
    -ignore-filename-regex='_deps|gtest' \
    -format=html \
    -output-dir=coverage_html
```

Finally, inspect the html output.

```bash
open coverage_html/index.html
```


### Build dependencies

* Required: [Eigen C++ template library for linear algebra](https://eigen.tuxfamily.org/index.php?title=Main_Page)
([MPLv2 licensed](https://www.mozilla.org/en-US/MPL/2.0/))
* Optional: [Mimalloc](https://microsoft.github.io/mimalloc/) ([MIT
  licensed](https://opensource.org/license/mit)). Can be used to improve
  allocator performance.
* Optional: [CLI11](https://cliutils.github.io/CLI11/) ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause)). Used by the example programs

### Test dependencies

* Required: [Google test](https://github.com/google/googletest) ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause))

### Developer dependencies

These are not managed by CMake and should be installed from your system package
manager (`brew`, `apt`, etc) as needed.

* clang-format
* include-what-you-use

## Building the Python library

The Python build uses a combination of CMake and
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/). The
only required runtime dependency is [numpy](https://numpy.org/),
though both
[BridgeStan](https://roualdes.us/bridgestan/latest/languages/python.html) and
[numba](https://numba.pydata.org/) will be used if installed.

The recommended way for developers to install the library is to run the
following command from the top-level directory of the repository.

```bash
pip install -e .
```

This will install the package in 'editable mode', which means local changes to
the Python or C++ files will be picked up and used when Python is restarted.

### Running the Python tests

We rely on a few extra packages for testing, these can be installed
with `pip install -e '.[test]'`. One of these is
[pytest](https://docs.pytest.org/en/stable/), which is used to actually launch
the test suite:

```bash
pytest python/ -v
```

## Building the Documentation

The documentation is built using
[Sphinx](https://www.sphinx-doc.org/en/master/index.html).

The top level organization of the API documentation is controlled
through `.rst` files (reStructuredText format) in 'docs/'
with `docs/index.rst` as the root.

The Python and C++ API doc are automatically generated from the
docstrings/comments in the respective source files. This requires
[Doxygen](https://www.doxygen.nl/) to be installed at the system level.


The other dependencies, including Sphinx itself, can be pip installed. The
recommended way to install these is to run the following command from the
top-level directory of the repo:

```sh
pip install '.[stan]' -r docs/requirements.txt
```

To build the documentation after the prerequisites are installed,
change directories to the `docs/` subfolder of the repository and run `make`
with the desired format, e.g. `html`:
```sh
cd docs/
make html
```
(if `make` is not installed, the second command is equivalent to
`sphinx-build -b html . _build/html`)

The above will output the documentation website in `_build/html`. Other valid
formats include `latexpdf`, which will require a LaTeX toolchain
installed.
