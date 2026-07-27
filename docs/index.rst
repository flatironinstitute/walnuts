
WALNUTS DOCUMENTATION
======================================================================

This is the documentation for Walnuts, a Markov chain Monte Carlo
(MCMC) sampler for differentiable target log densities.


C++ core
----------------------------------------------------------------------

Walnuts is implemented in multi-threaded C++20.

.. toctree::
   :maxdepth: 2

   cpp

Python interface
----------------------------------------------------------------------

Walnuts provides a Python API.

.. toctree::
   :maxdepth: 2

   py


The Python API accepts target log densities and gradients coded in
Python, including models coded in `NumPyro
<https://num.pyro.ai/en/latest/index.html>`__, `PyMC
<https://www.pymc.io/welcome.html>`__, `JAX
<https://docs.jax.dev/en/latest/>`__, or directly in Python, even with
foreign function calls.

`Stan <https://mc-stan.org>`__ models can be accessed directly at the C++ level through
`BridgeStan <https://roualdes.us/bridgestan/latest/>`__.


License
----------------------------------------------------------------------

Walnuts is distributed under the
`MIT License <https://opensource.org/license/mit>`__.

Stan and BridgeStan are distributed under the `BSD-3
<https://opensource.org/license/bsd-3-clause>`__ license. These
packages are only required to run models coded in Stan.


About the Walnuts Sampler
----------------------------------------------------------------------

`Walnuts <https://www.jmlr.org/beta/papers/v27/25-1452.html>`__ is a
Markov chain Monte Carlo (MCMC) sampler based the `no-U-turn sampler
<https://jmlr.org/beta/papers/v15/hoffman14a.html>`__ (Nuts), which in
turn is based on `Hamiltonian Monte Carlo
<https://arxiv.org/abs/1206.1901>`__ (HMC). Nuts
adds dynamic integration time selection to HMC. Walnuts introduces
dynamic step-size selection to deal with multi-scale target densities.

In addition to these sampling improvements over HMC, the Walnuts
implementation here departs from the Nuts implementation found in
Stan (and elsewhere) in several ways.

#. For estimating a mass matrix during warmup, Walnuts uses an online
   (iteration by iteration) version of `Nutpie
   <https://arxiv.org/abs/2603.18845v1>`__ warmup that takes a
   geometric mean of estimates based on the inverse covariance of
   draws and covariance of scores; the past is exponentially discounted on
   a diminishing schedule to follow Stan's geometrically increasing history
   lengths.

#. For estimating the maximum step size during warmup, Walnuts uses `Adam
   <https://arxiv.org/abs/1412.6980>`__ rather than dual averaging for
   stochastic gradient descent.

#. For running chains, Walnuts uses asynchronous threading and monitors
   progress through lock-free buffers in an additional thread, with
   automatic stopping when convergence is detected to within a
   specified threshold.

#. For posterior analysis of the resulting varying-length Markov
   chains, Walnuts provides estimators for posterior means, standard
   deviations, quantiles, R-hat, effective sample size, and Monte
   Carlo standard error.

.. #. There is an efficient binary output format in addition to a
      slow, but directly portable comma-separated value format.
