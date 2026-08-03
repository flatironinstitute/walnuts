# Contributing to `walnutpie`

Thank you for your interest in contributing to `walnutpie`.
Contributions are managed through Github
[issues](https://github.com/flatironinstitute/walnutpie/issues) and [pull
requests](https://github.com/flatironinstitute/walnutpie/pulls).

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


## C++ standard

We are currently using [C++20](https://cppreference.com/cpp/20).

We use `clang-format` for style standardization (see below).

## Building the C++ library

We use [CMake](https://cmake.org/) to manage dependencies and build
our C++ code.


The basic configuration is to run the following command from the
top-level `walnutpie` directory.

```sh
cmake <options> <repo_root>
```

Here, `<options>` is a sequence of CMake options and `<repo_root>` is
the root directory of the repository (where `CMakeLists.txt` is
found).

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

Unit tests are written using
[GoogleTest](https://google.github.io/googletest/).  We recommend
running the tests using `ctest`, which is distributed with CMake.

```sh
cmake --build build/ -j4
cd build/
ctest --output-on-failure
```

#### Test coverage reports

Test coverage reports can be generated if you are using a LLVM-based
C++ toolchain.  To test code coverage during testing, you will have to
specify the top-level `cmake` call to include
`DWALNUTPIE_COVERAGE=ON`.

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
licensed](https://opensource.org/license/bsd-3-clause)). Used only by
the example programs, not int he API itself

### Test dependencies

* Required: [GoogleTest](https://github.com/google/googletest) ([BSD-3
licensed](https://opensource.org/license/bsd-3-clause))

### Developer dependencies

These are not managed by CMake and should be installed from your system package
manager (`brew`, `apt`, etc) as needed.

* `clang-format`
* `include-what-you-use`

## Building the Python library

The Python build uses a combination of CMake and
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/).

The only required runtime dependency is

* [NumPy](https://numpy.org/)

The two optional dependencies are

* [BridgeStan](https://roualdes.us/bridgestan/latest/languages/python.html),
which can be used to connect to Stan models directly in C++.
* [Numba](https://numba.pydata.org/), which can be used to accelerate
models written directly in Python.

The Python library my be installed by running the following command
command from the top-level directory of the repository (the one with
`pyproject.toml` in it).

```bash
pip install -e .
```

This will install the package in editable mode, which means local
changes to the Python or C++ files will be picked up and used when
Python is restarted.

### Running the Python tests

The extra packages used only for testing can be installed with `pip
install -e '.[test]'`. To launch the Python tests, use
[pytest](https://docs.pytest.org/en/stable/),
which is downloaded as part of installing `.[test]`.

```bash
pytest python/ -v
```


## Documentation

### API documentation

The C++ and Python API documentation is generated from docstrings
encoded in the text files.  We use standard
[Doxygen](https://www.doxygen.nl/) style for C++ and [NumPy docstring
style](https://numpydoc.readthedocs.io/en/latest/format.html) for the
Python API.

### Additional documentation

Additional package documentation is coded in [reStructuredText
format](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
The top-level organization of the Python API documentation is
controlled through the reStructuredText files (suffix `.rst`) 'docs/'.
The root of the generated documentation is with `docs/index.rst` as
the root.

### Documented examples

Examples are automatically generated using the latest code through
[Jupyter notebooks](https://jupyter.org/).

### Building the documentation

The documentation is built using [Sphinx](https://www.sphinx-doc.org/).

The Python and C++ API doc are automatically generated from the
docstrings/comments in the respective source files. This requires
[Doxygen](https://www.doxygen.nl/) and [Pandoc](https://pandoc.org/)
to be installed at the system level.


The other dependencies, including Sphinx itself, can be pip installed. The
recommended way to install these is to run the following command from the
top-level directory of the repo:

```sh
pip install '.[stan]' -r docs/requirements.txt
```

To build the HTML documentation after the prerequisites are installed,
change directories to the `docs/` subfolder of the repository (the one
with `index.rst` in it)
```sh
cd docs/
```
and then run Make with target `html`.
```sh
make html
```
If `make` is not installed in your system, you can replace `make html` with
```sh
sphinx-build -b html . _build/html
```
If the build is successful, the root of the documentation will be
found in `_build/html/index.html` (in the `docs` directory).

To build a PDF document, use
```sh
make latexpdf
```
Making the PDF document requires a LaTeX toolchain including the
command `xelatex` to be installed at the system level.  `xelatex` is
required to deal with embedded unicode. If the uild is successful, the
PDF documentation will be found in `_build/latex/walnutpie.pdf` (in
the `docs` directory).
