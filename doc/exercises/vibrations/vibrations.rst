================================================
Vibrational modes of the H\ :sub:`2`\ O molecule
================================================

Density functional theory can be used to calculate vibrational frequencies of
molecules, e.g. either in the gas phase or on a surface. These results can be
compared to experimental output, e.g. from IR-spectroscopy, and they can be
used to figure out how a molecule is bound to the surface. In this example we
will calculate the vibrational frequencies for a water molecule.

* For a simple molecule, like CO, there is only one stretching mode. How would
  you calculate the vibrational frequency of this mode?

* For a general molecule with N atoms, how many modes are there? How many of
  them are vibrational modes? How would you do a calculation for the vibrational
  modes? Describe in detail which steps have to be performed.

* Make a script where a H\ :sub:`2`\ O molecule is relaxed to its equilibrium
  position.  It can be done like this:
  :svn:`~doc/exercises/vibrations/h2o.py`.  An alternative is to
  use the MP2 structures from ASE's molecular database::

    from ase.structure import molecule
    h2o = molecule('H2O')
    h2o.center(vacuum=3.5)

* Copy the file :svn:`~doc/exercises/vibrations/H2O_vib.py`
  to your area and try to understand what it does.

* Run the script and look at the output frequencies. Compare them to
  literature values, which are 1595cm\ :sup:`-1` for the bending
  mode, 3657cm\ :sup:`-1` for the symmetric stretching mode and
  3756cm\ :sup:`-1` for the anti-symmetric stretching mode.  How good
  is the accuracy and what are possible error sources?

* Now we want to look at the modes to see how the atoms move. For this
  we use the files :file:`vib.{?}.traj` where :file:`{?}` is the number of the
  mode counted in the order they are printed out. You can look at
  these trajectories with the :command:`ase-gui` command - click :guilabel:`Play` 
  to play the movie. Do they look like you expected and what would you have
  expected (you may have learned something about symmetry groups at
  one point)?  Did you assign the modes correctly in the previous question?
