Python API
==========


Sampling functions
------------------


.. autofunction:: walnuts.walnuts_pyfunc
.. autofunction:: walnuts.walnuts_stan


Output Classes
______________

.. autoclass:: walnuts.pyfunc::WalnutsOutputArray
   :members:
   :show-inheritance:

.. autoclass:: walnuts.stan::StanOutput
   :members:
   :inherited-members:
   :show-inheritance:
   :special-members: __getitem__

.. autoclass:: walnuts.stan::StanOutputBase

.. autoclass:: walnuts.util::WarmupInfo
   :members:

Summary
-------

.. autoclass:: walnuts.Summarizer
   :members:

.. autofunction:: walnuts.ess
.. autofunction:: walnuts.r_hat
.. autofunction:: walnuts.mcse
.. autofunction:: walnuts.mean
.. autofunction:: walnuts.variance
.. autofunction:: walnuts.standard_deviation
