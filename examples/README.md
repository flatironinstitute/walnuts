# BENCHMARKS

## Stationarity of adaptive WALNUTS

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


## Warmup for adaptive WALNUTS

This test visualizes how quickly adaptive WALNUTS is able to adapt the
inverse mass matrix and step size.  It uses a 200-dimensional normal
with no correlation and variances equal to the index sqared, i.e., 1,
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


## Gradients until within error bound

For Hamiltonian Monte Carlo samples from a target density `p(theta)`
(e.g., a Bayesian posterior), the evaluation statistic is the number
of gradient evaluations required before the standardized error for
estimating the expectation of each component of `X` and `X^2` is below
a threshold (e.g., 0.1), for `X = theta, log p(theta)`, where `theta`
is a vector of sampled values.  Here, `theta` contains constrained
Bayesian model parameters.

Standardized error is scaled as the number of standard devations an  
estimate is from the mean. If `hatY` is an estimate of the expectation
of a random variable `Y`, the standardized error is `(hatY - E[Y]) /
sd[Y]`, where `E[Y]` is the expectation of `Y` and `sd[Y]` is the
standard deviation of `Y`.


### Directory structure

The directory `dir = examples/models` will have a model subdirectory
`<model>` for each model being evaluated, e.g.,
`examples/models/ill-cond-normal`.  Each model sudirectory holds two
user-generated files:

* Stan program: `<model>.stan`
* Data (JSON format): `<model>-data.json`


### Step 1. Generarte reference moments

To estimate the reference first, second, and fourth moments, run:

```bash
cd examples
python reference-moments.py <model> [target-ESS]
```

The arguments are:

* `model`: The name of the model subdirectory. 
* `target-ESS`: The target effective sample size.

This may take a long time to run for complex models or high effective
sample sizes.  Reference moments will be generated in:

* Reference moments: `<model>-moments.json`


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
  
If successful, the result will be generated in:

* NUTS results: `<model>-nuts-gradients.json`

The JSON file defines a single key `gradients` with a sequence of
integer values of length `trials` for the number of gradients required
before estimation error in every first and second moment is below
threshold.


### Step 3. Measure WALNUTS gradients to error threshold

*Step 3.1*: To generate the WALNUTS draws:

To compile the model into a shared object using 
[`bridgestan`](https://github.com/roualdes/bridgestan), which must be
installed in Python.

```python
import bridgestan as bs
bs.compile_model(stan_file="<model>.stan)")
```

This will compile the Stan program into a binary shared object file:

* `<model>.so`


*Step 3.2*: To compile and run the C++ evaluation:

```bash
cd build
make eval_stan
./eval_stan <model> <iter_warmup> <iter_sampling>
```

* `model`: The name of the model subdirectory. 
* `iter_warmup`: The number of warmup iterations. 
* `iter_sampling`: The number of sampling iterations.
* `num_trials`: The number of repetitions of sampling to run.

The MCMC draws will be generated in:

* WALNUTS draws: `<model>-walnuts-draws.csv`


*Step 3.3*: To generate the number of gradients required, run:

```
cd examples
python walnuts-gradients.py <model> <max_error>
```

where

* `model`: The name of the model subdirectory. 
* `max_error`: The maximum standardized error allowed in first or 
  second moments. 

### Step 4. Plot NUTS and WALNUTS gradients to error threshold
