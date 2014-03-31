.. _convergence:

==================
Convergence Issues
==================

Here you find a list of suggestions that should be considered when
encountering convergence problems:

* Try to use default parameters for the calculator. Simple but
  sometimes useful!

* Remember that for Gamma Point calculations the :ref:`Fermi
  temperature <manual_occ>` is set to zero by default, even if you
  use periodic boundary conditions. In the case of metallic systems
  you might want to specify a finite Fermi temperature.
  Remember to check the convergence of the results with respect to
  the finite Fermi temperature!

Look at the :ref:`scf_conv_eval` page in order to select the parameters
that may be suitable for your system. Specifically:

* If you are specifying the :ref:`number of bands <manual_nbands>`
  manually, try to increase the number of empty states. You might also
  let gpaw choose the default number, which is in general large
  enough. A symptom of having an insufficient number of empty bands is
  large variations in the number of SCF iterations with different
  number of MPI tasks. It is reasonable to need about 10-15% additional bands
  to properly converge the default ``rmm-diis`` eigensolver.

* Try to use a less aggressive :ref:`mixing <manual_mixer>`, i.e. a
  smaller mixing parameter. For example a typical mixer for a metallic
  system may be ``mixer=Mixer(0.05, 5, weight=100.0)``. Smaller number
  of diis mixing steps may work better sometimes ``mixer=Mixer(0.05,
  2, weight=100.0)``.  For spin polarised systems you can use either
  ``mixer=MixerSum()`` or ``mixer=MixerDif()`` using the same options
  as with ``mixer=Mixer()``.

* Sometimes for metallic systems of large dimensions (thick slabs or
  large clusters), one can have the situation that the wave functions
  converge nicely, but the density does not.  For such cases it can
  help to solve the Poisson equation more accurately between each SCF
  step.  Try something like ``poissonsolver=PoissonSolver(eps=1e-12)``.

* The initial guess for the electron density is always calculated
  using the LCAO scheme, with a default single-zeta basis, i.e. one
  orbital for each valence electron. You can try to make a better
  initial guess by enlarging the :ref:`manual_basis`. This can be done
  by setting ``basis='szp'`` if you want to use a
  single-zeta-polarized basis. Note that you first need to generate
  the basis file, as described in :ref:`LCAO mode <lcao>`. As of GPAW
  0.7.2 or later, it is also possible to use ``basis='szp(dzp)'`` to extract
  the single-zeta polarization basis set from the double-zeta
  polarization basis sets that is available in the latest Setups. 
 
* For spin polarised systems try to play with initial magnetic moments.

* If all else fails, try changing the :ref:`eigensolver <manual_eigensolver>`.
  The default, ``rmm-diis``, is fast, but can sometimes have
  poor convergence properties. The conjugate gradient, ``cg``, solver
  might be more stable, but slower. For very difficult systems you may
  need to use some of the above tricks also for ``cg``.
  Are you sure you explored all these options with ``rmm-diis``?
