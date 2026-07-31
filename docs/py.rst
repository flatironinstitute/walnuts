Python API
==========


Sampling functions
------------------


.. autofunction:: walnutpie.walnuts_pyfunc
.. autofunction:: walnutpie.walnuts_stan


Output Classes
______________

.. autoclass:: walnutpie.pyfunc::WalnutsOutputArray
   :members:
   :show-inheritance:

.. autoclass:: walnutpie.stan::StanOutput
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __getitem__

.. autoclass:: walnutpie.stan::StanOutputBase

.. autoclass:: walnutpie.util::WarmupInfo
   :members:

Summary
-------

.. autoclass:: walnutpie.Summarizer
   :members:

.. autofunction:: walnutpie.ess
.. autofunction:: walnutpie.r_hat
.. autofunction:: walnutpie.mcse
.. autofunction:: walnutpie.mean
.. autofunction:: walnutpie.variance
.. autofunction:: walnutpie.standard_deviation
