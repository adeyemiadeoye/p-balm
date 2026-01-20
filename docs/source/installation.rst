Installation
============

Install **pbalm** using pip:

.. code-block:: bash

   python3 -m pip install pbalm

JAX Configuration
-----------------

For optimal performance, you may want to configure JAX settings:

.. code-block:: python

   import jax
   
   # To use CPU backend
   jax.config.update('jax_platform_name', 'cpu')
   
   # Enable 64-bit precision (recommended)
   jax.config.update("jax_enable_x64", True)

.. note::

   Enabling 64-bit precision is recommended for better numerical accuracy,
   especially for problems with tight convergence tolerances.
