C++ API
=======

Sampling functionality
----------------------

Top-level call
______________

This function will spawn threads to perform end-to-end sampling

.. doxygenfunction:: walnuts::walnuts


Iterator-style samplers
_______________________

These classes implement WALNUTS in an iteration-per-call style iterator.

.. doxygenclass:: walnuts::WalnutsSampler
   :members:
.. doxygenclass:: walnuts::AdaptiveWalnuts
   :members:

Configuration
-------------

The following classes (and their builders) are used to configure WALNUTS

.. doxygenclass:: walnuts::WalnutsConfig
   :members:

.. doxygenclass:: walnuts::InitConfigBuilder
   :members:
.. doxygenclass:: walnuts::WarmupConfigBuilder
   :members:
.. doxygenclass:: walnuts::SamplingConfigBuilder
   :members:


.. doxygenclass:: walnuts::InitConfig
.. doxygenclass:: walnuts::WarmupConfig
.. doxygenclass:: walnuts::SamplingConfig


Concepts
--------

The following concepts describe the types expected by WALNUTS.

.. doxygenconcept:: walnuts::LogpGrad
.. doxygenconcept:: walnuts::ChainHandler
.. doxygenconcept:: walnuts::SampleHandler
.. doxygenconcept:: walnuts::GlobalHandler
.. doxygenconcept:: walnuts::InterruptCallback
