.. _sisu:

============================
sisu.csc.fi  (Cray XC30) 
============================

.. note::
   The current libsci library in the system gives incorrect results with
   certain input data. Problem can be circumveted by using always ScaLAPACK.


.. note::
   These instructions are up-to-date as of February 26th 2013.

GPAW
====

These instructions for GPAW installation use the Python provided by Cray.
First, change the default compiler environment to GNU and load the HDF5 
module::

  module swap PrgEnv-cray PrgEnv-gnu
  module load cray-hdf5-parallel

GPAW can be build with a minimal ``customize.py``

.. literalinclude:: customize_sisu.py

Then just run the setup.py script (the system provided Python needs an empty
--prefix argument when using --home)::

  python setup.py install --home=... --prefix=''

