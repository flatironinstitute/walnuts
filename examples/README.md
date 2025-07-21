# BENCHMARKS

## Stationarity of adaptive WALNUTS

This test ensures that there is no bias in the sampler. It works by
taking 10M draws from adaptive WALNUTS for a 20-dimensional normal
target with no correlation and variances equal to the index squared,
i.e., 1, 4, 9, ..., 400. The following code runs the benchmark

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



