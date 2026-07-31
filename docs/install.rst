Installation
=============


``walnutpie`` is available `on PyPI <https://pypi.org/project/walnutpie/>`__
with pre-compiled wheels provided for major platforms.

.. code-block:: shell

   pip install walnutpie

If you would like to use ``walnutpie`` with Stan models, you should additionally
install BridgeStan. See :external+bridgestan:doc:`getting-started` in the
BridgeStan documentation for instructions.


Installing from source
----------------------

To install ``walnutpie`` from source, ensure you have a
C++20-compatible compiler installed and run

.. code-block:: shell

   pip install git+https://github.com/flatironinstitute/walnuts.git#egg=walnutpie
