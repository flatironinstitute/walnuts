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


