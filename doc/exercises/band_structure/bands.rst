.. _band_exercise:

==============
Band structure
==============

Band diagrams are usefull analysis tools. Read
:svn:`~doc/exercises/band_structure/Na_band.py` and try to
understand what it does, then use it to construct the band diagram of
bulk Na.  Read
:svn:`~doc/exercises/band_structure/plot_band.py` and try
to understand it, then use it to plot the band diagram.

Modify the first script to calculate the band diagram of Silicon.
Note that Si has a different crystal structure (diamond), either set
it up manually as for Na, or use the :ase:`ase.lattice.bulk <ase/structure.html>` function.

Compare the Si band diagram to the band diagram below (adapted from
Cohen and Chelikowsky: "Electronic Structure and Optical Properties of
Semiconductors" Solid-State Sciences 75, Springer-Verlag 1988).  How
does this correspond to the DOS calculation in the previous exercise?

.. figure:: silicon_banddiagram.png
   :width: 600 px
   :align: center

For a description of the symmetry labels of the Brillouin zone; see
the figure below.

.. figure:: ../../_static/bz-all.png
   :width: 600 px
   :align: center
