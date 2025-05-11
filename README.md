# WALNUTS in C++

First, we are building and integrating NUTS.  Second, we will extend to WALNUTS.

## Running NUTS sampler

First, you need to download the Eigen 3.4.0 source and place it in a directory `walnuts_cpp/lib/eigen-3.4.0`.

#### Compiling and running

```
$ cd walnuts_cpp
$ clang++ -std=c++17 -O3 -I lib/eigen-3.4.0 test.cpp -o test
$ ./test
```

To compile with GCC, just use `g++` instead of `clang++` in the compilation command above.

#### Formatting

```
$ cd walnuts_cpp
$ clang-format clang-format -i -style=LLVM nuts.hpp test.cpp
```
