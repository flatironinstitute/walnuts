# Models for evaluation

The directory `examples/models` has one subdirectory per model/data combination.  Each of these directories should contain exactly three files:

* `XXX.stan`: Stan program for model XXX.
* `XXX-data.json`: JSON data file for model XXX.
* `XXX-moments.json`: The moments, with keys `first`, `second`, `third`, and `fourth`, along with `ess_first`, `ess_second`, `ess_third`, and `ess_fourth`.  The ordering of variables within the arrays is the unnormalized log density variable `lp__` followed by the parameters in the order they were declared in the Stan program (i.e., the same order as they appear in `summary()` in CmdStanPy).


## Generating reference moments

Given a subdirectory `examples/models/XXX` that contains files `XXX.stan` file and `XXX-data.json` file, the program `example/reference-moments.py` can be used to generate reference moments in `examples/models/XXX-moments.json`.  The minimimum effective sample size and file can be specified within `reference-moments.py` for now.

```bash
cd examples
python reference-moments.py
```
