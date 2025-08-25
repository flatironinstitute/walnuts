# BENCHMARKS

## I.  Stationarity of adaptive WALNUTS

This test ensures that there is no bias in the sampler. It works by
taking 10M draws from adaptive WALNUTS for a 20-dimensional normal
target with no correlation and variances equal to the index squared,
i.e., 1, 4, 9, ..., 400. The following code runs the benchmark.

```sh 
cd build 
make walnuts-stationarity 
./walnuts-stationarity 
python ../examples/walnuts-stationarity.py 
open walnuts-stationarity.jpg 
```

1.  The program `walnuts-stationarity` will write a CSV file of draws to
`walnuts-stationarity.csv`. This takes about a minute.

2.  The Python program `examples/walnuts-stationarity.py` reads the CSV
file and generates the plot `build/walnuts-stationarity.jpg`. This
takes about minute, too.

The source code for these tests:

* `examples/walnuts-stationarity.cpp`
* `examples/walnuts-staionarity.py`


## II. Warmup for adaptive WALNUTS

This test visualizes how quickly adaptive WALNUTS is able to adapt the
inverse mass matrix and step size.  It uses a 200-dimensional normal
with no correlation and variances equal to the index squared, i.e., 1,
4, 9, ..., 40000.  The following code runs the benchmark.

```sh 
cd build 
make walnuts-warmup
./walnuts-warmup
python ../examples/walnuts-warmup.py 
open walnuts-warmup-inv-mass.jpg 
```

This generates two output files:

* `build/walnuts-warmup-inv-mass.jpg` for the inverse
mass matrix adaptation, and 
* `build/walnuts-warmup-step.jpg` for the step size
adaptation. 

All of this code runs very quickly.

For comparison, we also provide the default warmup values for inverse
mass matrix and step size from Stan.  Because Stan doesn't warm up
continuously, these values report the state at the end of the warmup
phase when the warmup phase is set to the specified number of
iterations. This is Stan's default behavior, but Stan can be
configured for different blocking during warmup.

```sh
cd build
python3 ../examples/stan-warmup.py
```

This generates an output plot:

* `build/stan-warmup-inv-mass.jpg` for the inverse mass matrix
  adaptation. 

The source code for these tests: 

* `examples/walnuts-warmup.cpp`
* `examples/walnuts-warmup.py`
* `examples/stan-warmup.py`


## III. Gradients until within error bound

For Hamiltonian Monte Carlo samples from a target density `p(theta)`
(e.g., a Bayesian posterior), a natural measurement is the number of
gradient evaluations required before the standardized error for
estimating the expectation of each component of `X` and `X^2` is below
a threshold (e.g., 0.1).  Throughout, `X = theta, log p(theta)`, where
`theta` is a vector of sampled parameters and `log p(theta | y)` is the
posterior log density (up to a fixed constant).

Standardized error is scaled by the standard deviation, giving a
standardized standard error scaling error as a number of standard
deviations from the mean (i.e., a *Z*-score).
If `hatY` is an estimate of the expectation
of a random variable `Y`, the standardized error is `(hatY - E[Y]) /
sd[Y]`, where `E[Y]` is the expectation of `Y` and `sd[Y]` is the
standard deviation of `Y`.

Because `sd[Y] = sqrt(var[Y])` and `var[Y] = E[Y^2] - E[Y]^2`, the
second moment `Y^2` is required to standardize estimates of `E[Y]` and
hence the fourth moment `Y^4` is required to standardize estimates of
`E[Y^2]`.


### Directory structure

The directory `dir = examples/models` will have a model subdirectory
`<model>` for each model being evaluated, e.g.,
`examples/models/ill-cond-normal`.  Each model subdirectory should
contain two user-generated files following the naming convention:

* Stan program: `<model>.stan`
* Data for program: `<model>-data.json`


### Step 1. Generate reference moments

To estimate the reference moments (first, second, and fourth), run:

```bash
cd examples
python reference-moments.py <model> [target-ESS]
```

The arguments are:

* `model`: The name of the model subdirectory. 
* `target-ESS`: The target effective sample size.

The estimation is done with Stan's implementation of NUTS with default
tuning settings.  It may take a long time to run for complex models or
high effective sample sizes.  Reference moments will be generated in:

* Reference moments: `<model>-moments.json`

There are three keys, `first`, `second`, and `fourth`, with array
values in scientific notation to 16 decimal places.


*Example*:

```bash
python reference-moments.py ill-normal 100_000

head models/ill-normal/ill-normal-moments.json
```

This takes a minute or two to run (1m on a 2022 M2 Macbook Air).

### Step 2.  Measure NUTS gradients to error threshold

To generate the NUTS measurements:

```bash
python eval-nuts.py <model> <max_error> <trials> <iter_warmup> <iter_sampling>
```

* `model`: The name of the model subdirectory. 
* `max_error`: The maximum standardized error allowed in first or 
  second moments. 
* `trials`: The number of times to repeat the experiments. 
* `iter_warmup`: The number of warmup iterations. 
* `iter_sampling`: The number of sampling iterations. 
  
The result will be generated in:

* NUTS results: `<model>-nuts-gradients.json`

The JSON file defines a single key `gradients` with a sequence of
integer values of size `trials` for the number of gradients required
before estimation error in every first and second moment is below
the specified `max_error`.  If `iter_sampling` is too low, it can
be raised until the `max_error` threshold is satisfied.

To plot a histogram of the results, run 

```bash 
python plot-grads <model>
```

*Example*:

```bash
python eval-nuts.py ill-normal 0.1 128 256 5000

head models/ill-normal/ill-normal-nuts-gradients.csv
```

This takes a few minutes to run (2m on a 2022 M2 Macbook Air). 

```bash
python plot-grads ill-normal
```


This will display a plot and save it as a JPG: 
`examples/models/<model>-nuts-grads.jpg`. 


### Step 3. Measure WALNUTS gradients to error threshold

**Step 3.1**: Compile Stan program.

Start from the directory in which the Stan program resides,

```bash
cd examples/models/<model>
```

To compile the model into a shared object using 
[`bridgestan`](https://github.com/roualdes/bridgestan),

```python
import bridgestan as bs
bs.compile_model("<model>.stan")
```

The result is a binary shared object file:

* `<model>_model.so`


**Step 3.2**: Make the C++ evaluation.

Once only, run the setup for CMake from the top-level repository
directory `walnuts`:

```bash
cd walnuts

cmake . -B ./build -DCMAKE_BUILD_TYPE=Release
```

See the top-level [`README`](../) for more information on configuring CMake.

Then `cd` into the `build` directory

```bash
cd build
```

and run the evaluation

```bash
make eval_walnuts
./eval_walnuts <dir> <model> <seed> <iter_warmup> <iter_sampling> <trials>
```

* `dir`: The name of the directory containing the model subdirectory. 
* `model`: The name of the model subdirectory. 
* `seed`: Seed for random number generation.
* `iter_warmup`: The number of warmup iterations. 
* `iter_sampling`: The number of sampling iterations. 
* `trials`: The number of repetitions to run. 

The MCMC draws will be generated in:

* WALNUTS draws: `<model>-walnuts-draws-<num>.csv`

for `<num>` in `{0, ..., trials - 1}`.


**Step 3.3**: Generate gradients CSV

First `cd` into the `examples` directory,

```bash
cd examples
```

then run the gradient extractor,

```bash
python walnuts-gradients-to-error.py <dir> <model> <trials> <max_error>
```

* `dir`: The name of the directory containing the model subdirectory. 
* `model`: The name of the model subdirectory. 
* `trials`: The number of repetitions that were run with walnuts.
* `max_error`: The maximum standardized error allowed in first or 
  second moments. 
  
The number of gradients per trial will be generated in CSV format in
the file: 

* WALNUTS gradients: `<model>-walnuts-gradients.csv`


### Step 4. Plot NUTS and WALNUTS gradients to error threshold

To plot NUTS vs. WALNUTS performance on a gradients until first 
and second moments of log density and parameters are below the
error threshold, first change into the examples directory,

```bash
cd examples
```

and then

```bash
python plot-grads.py <model>
```

* `model`: The name of the model subdirectory. 

This will produce a JPG plot in

* Performance plot: `examples/<model>/<model>-nuts-vs-walnuts-grads.jpg`
