# Models for evaluation

The directory `examples/models` has one subdirectory per model/data combination.  Each of these directories should contain exactly three files:

* `XXX.stan`: Stan program for model XXX.
* `XXX-data.json`: JSON data file for model XXX.
* `XXX-moments.json`: The moments, with keys `moment1`, `moment2`, `moment4`, for the first, second, and fourth moments, `ess1`, `ess2`, and `ess4` for the effective sample sizes of the rirst, second, and fourth moments.  The ordering of variables within the arrays is the order of the variables in the Stan declaration followed the variable `lp__` from Stan representing the unnormalized log density.


## Generating reference moments

Given a subdirectory `examples/models/XXX` that contains files `XXX.stan` file and `XXX-data.json` file, the program `example/reference-moments.py` can be used to generate reference moments in `examples/models/XXX-moments.json` based on a mimimum effective sample size of 100K.
