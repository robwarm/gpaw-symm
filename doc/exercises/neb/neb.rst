================================
Nudged elastic band calculations
================================

Self-diffusion on the Al(110) surface
-------------------------------------

.. image:: Al110slab.png
   :height: 270 px
   :alt: Al(110) surface
   :align: right

In this exercise, we will find minimum energy paths and transition
states using the :ase:`Nudged Elastic Band <ase/neb.html>` method.

Take a look at the Al(110) surface shown in the picture on the right.
The red atom represents an Al adatom that can move around on the surface.
The adatom can jump along the rows or across the rows.

* Which of the two jumps do you think will have the largest energy
  barrier?

The template script :svn:`~doc/exercises/neb/neb1.py` will
find the minimum energy path for a jump along the rows.  Read,
understand, and run the script.

* Make sure you understand what is going on (make a good sketch of the
  110 surface).

* What is the energy barrier?

* Copy the script to :file:`neb2.py` and modify it to find the barrier for
  diffusion across one of the rows.  What is the barrier for this
  process?

* Can you think of a third type of diffusion process?  Hint: it is
  called an exchange process.  Find the barrier for this process, and
  compare the energy barrier with the two other ones. (If you give up look at :svn:`~doc/exercises/neb/neb3.py`)

.. hint::

  When opening a trajectory in :program:`ase-gui` with calculated energies, the
  default plot window shows the energy versus frame number.  To get a
  better feel of the energy barrier in an NEB calculation; choose
  :menuselection:`Tools --> NEB`. This will give a smooth curve
  of the energy as a
  function of the NEB path length, with the slope at each point
  estimated from the force.
