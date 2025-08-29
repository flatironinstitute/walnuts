# WALNUTS COMMAND-LINE INTERFACE

This directory, `cli`, contains a C++ implementation of a very simple
one-file command-line interface to WALNUTS based on the
[`cxxopts`](https://github.com/jarro2783/cxxopts) library. The
implementation is in [`walnuts.cpp`](walnuts.cpp).

## Basic usage

Here's the basic usage from the command itself.  Start from the
`build` directory (build instructions below).  

```bash
cd build
```

Running the `help` command

```bash
./walnuts_cli --help
```

produces the output

```bash
WALNUTS command-line interface
Usage:
  walnuts_cli [OPTION...]

  -h, --help                    Show help
      --stan <path>             Compiled Stan .so file
      --data <path>             Path to data .json file (default: "")
      --out <path>              Output sample .csv file (default: out.csv)
      --n_warmup <int>          Warmup iterations (default: 128)
      --n_sample <int>          Sampling iterations (default: 128)
      --seed <uint>             Random number generator seed (default: 42)
      --mass_count <double>     Pseudocount of initial mass matrix 
                                (default: 1.1)
      --mass_offset <double>    Pseudoposition after initial mass matrix 
                                (default: 1.1)
      --mass_smooth <double>    Additive smoothing to mass matrix (default: 
                                1e-5)
      --step_init <double>      Initial step size (default: 1.0)
      --accept_target <double>  Target Metropolis accept probability 
                                (default: 0.8)
      --step_offset <double>    Pseudoposition after initial step size 
                                (default: 5.0)
      --learning_rate <double>  Step size learning rate, higher is slower 
                                (default: 1.5)
      --decay_rate <double>     Step size decay of history rate for 
                                averaging (default: 0.05)
      --max_error <double>      Maximum absolute energy error allowed in 
                                leapfrog step (default: 0.5)
      --max_nuts_depth <int>    Maximum number of trajectory doublings in 
                                NUTS (default: 8)
      --max_step_depth <int>    Maximum number of step size halvings in 
                                WALNUTS (default: 8)
```

This command requires a path to a compiled Stan shared object (`.so`)
file. The `data` value should be formatted for Stan as a dictionary
with keys for each data variable (matrices are column major, arrays
are row major).  The `out` file is written in comma-separated-value
format.

To simulate vanilla NUTS without WALNUTS-style local stepsize
adaptation, set max_error to a very high value, e.g., `--max_error
1000000`. This will typically be faster in situations where the
curvature is fixed (i.e., the Hessian does not vary across the
posterior), as in roughly normal target densities.


To run a single Markov chain to produce a posterior sample from a
model coded in a Stan program `foo.stan` with data `foo-data.json`:

```bash
./walnuts_cli stan=foo.stan data=foo-data.json out=foo-sample.csv
```

The results can be inspected to make sure they define the appropriate
variables

```bash
head foo-sample.csv
```

Otherwise, this `.csv` file may be read into a data processing
language and posterior analysis run. 

* Python: [ArviZ](https://python.arviz.org/en/stable/)
* R: [posterior](https://mc-stan.org/posterior/articles/posterior.html)

As sampling is running or after, the results can be monitored online
with the following browser-based GUI.

* [MCMC Monitor](https://github.com/flatironinstitute/mcmc-monitor)


## Installation

The top-level directory contains a `CMakelists.txt` file that provides
build and dependency management.  This should be done in two steps:

First, from the top-level directory,

```bash
cd walnuts
```

configure the make process:

```bash
cmake . -B ./build -DCMAKE_BUILD_TYPE=Release
```

Second, change into the `build` directory,

```bash
cd build
```

and then build the command-line interface.

```bash
make walnuts_cli
```

You can then run from the `build` directory as shown above. 

CMake will fetch most of the dependencies. Before running, this
requires

* CMake on the path
* a C++17-compatible toolchain on the path

For use, the easiest way to compile Stan models to object code is with

* [`bridgestan`](https://roualdes.us/bridgestan/latest/)

Each of the BridgeStan interfaces provides a way to compile a Stan
model to a shared object (`.so`) file.  For example, in Python,
it's

```python
import bridgestan as bs
bs.compile_model(stan_file)
```

