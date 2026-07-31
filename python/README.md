# walnutpie - the Within-orbit Adaptive Leapfrog No-U-Turn Sampler

## Documentation

Documentation for `walnutpie` can be found on [Github Pages](https://flatironinstitute.github.io/walnuts/latest/).

## Basic usage

```python
import walnutpie as wp


def logp(x):
    # your code here
    lp = ...
    gradient = ...
    return lp, gradient


draws = wp.walnuts_pyfunc(logp, num_params=10)

print(wp.ess(draws))
```

## Prior work

The design of this library is heavily influenced by both
[TinyStan](https://github.com/WardBrian/tinystan) and [Nutpie](https://github.com/pymc-devs/nutpie).
