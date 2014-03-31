.. _lrtddft:

=====================
Linear response TDDFT
=====================

Ground state
============

The linear response TDDFT calculation needs a converged ground state calculation with a set of unoccupied states. The standard eigensolver 'rmm-diis' should not be used for the calculation of unoccupied states, better use 'dav' or 'cg':

.. literalinclude:: Be_gs_8bands.py

Calculating the Omega Matrix
============================

The next step is to calculate the Omega Matrix from the ground state orbitals:

.. literalinclude:: Be_8bands_lrtddft.py

alternatively one can also restrict the number of transitions by their energy:

.. literalinclude:: Be_8bands_lrtddft_dE.py

Note, that parallelization over spin does not work here. As a workaround,
domain decomposition only (``parallel={'domain': world.size}``, 
see :ref:`manual_parsize_domain`) 
has to be used for spin polarised 
calculations in parallel.

Extracting the spectrum
=======================

The dipole spectrum can be evaluated from the Omega matrix and written to a file::

  from gpaw.lrtddft import LrTDDFT
  from gpaw.lrtddft import photoabsorption_spectrum

  lr = LrTDDFT('lr.dat.gz')
  lr.diagonalize()
  # write the spectrum to the data file
  photoabsorption_spectrum(lr, 'spectrum_w.05eV.dat', # data file name
                           width=0.05)                # width in eV

Testing convergence
===================

You can test the convergence of the Kohn-Sham transition basis size by restricting
the basis in the diagonalisation step, e.g.::

  from gpaw.lrtddft import LrTDDFT 

  lr = LrTDDFT('lr.dat.gz')
  lr.diagonalize(energy_range=2.)

This can be automated by using the check_convergence function::

  from gpaw.lrtddft import LrTDDFT
  from gpaw.lrtddft.convergence import check_convergence

  lr = LrTDDFT('lr.dat.gz')
  check_convergence(lr,
                    'linear_response',
                    'my plot title',
                     dE=.2,
		     emax=6.)

which will create a directory 'linear_response'. In this directory there will be a
file 'conv.gpl' for gnuplot that compares the spectra varying the basis size.

Analysing the transitions
=========================

The single transitions (or a list of transitions) can be analysed as follows 
(output printed)::

  from gpaw.lrtddft import LrTDDFT

  lr = LrTDDFT('lr.dat.gz')
  lr.diagonalize()

  # analyse transition 1
  lr.analyse(1)

  # analyse transition 0-10
  lr.analyse(range(11))

Relaxation in the excited state
===============================

This example shows how to relax in the B excited state of the sodium dimer::

  from ase import Atom, io, optimize
  from gpaw import GPAW, FermiDirac
  from gpaw.cluster import Cluster
  from gpaw.lrtddft import LrTDDFT
  from gpaw.lrtddft.excited_state import ExcitedState

  box = 5.     # box dimension
  h = 0.25     # grid spacing
  width = 0.01 # Fermi width
  nbands = 6   # bands in GS calculation
  nconv = 4    # bands in GS calculation to converge
  R = 2.99     # starting distance
  iex = 1      # excited state index
  d = 0.01     # step for numerical force evaluation
  exc = 'LDA'  # xc for the linear response TDDFT kernel

  s = Cluster([Atom('Na'), Atom('Na', [0, 0, R])])
  s.minimal_box(box, h=h)

  c = GPAW(h=h, nbands=nbands, eigensolver='cg',
           occupations=FermiDirac(width=width),
           convergence={'bands':nconv})
  c.calculate(s)
  lr = LrTDDFT(c, xc=exc, eps=0.1, jend=nconv-1)

  ex = ExcitedState(lr, iex, d=d)
  s.set_calculator(ex)

  ftraj='relax_ex' + str(iex)
  ftraj += '_box' + str(box) + '_h' + str(h)
  ftraj += '_d' + str(d) + '.traj'
  traj = io.PickleTrajectory(ftraj, 'w', s)
  dyn = optimize.FIRE(s)
  dyn.attach(traj.write)
  dyn.run(fmax=0.05)


Quick reference
===============

Parameters for LrTDDFT:

================  ==============  ===================  ========================================
keyword           type            default value        description
================  ==============  ===================  ========================================
``calculator``    ``GPAW``                             Calculator object of ground state
                                                       calculation
``filename``      ``string``                           read the state of LrTDDFT calculation 
                                                       (i.e. omega matrix, excitations)
                                                       from ``filename``  
``istart``        ``int``         0                    first occupied state to consider
``jend``          ``int``         number of bands      last unoccupied state to consider
``energy_range``  ``float``       None                 Energy range to consider in the involved
                                                       Kohn-Sham orbitals (replaces [istart,jend])
``nspins``        ``int``         1                    number of excited state spins, i.e.
                                                       singlet-triplet transitions are 
                                                       calculated with ``nspins=2``. Effective
                                                       only if ground state is spin-compensated
``xc``            ``string``      xc of calculator     Exchange-correlation for LrTDDFT, can 
                                                       differ from ground state value 
``eps``           ``float``       0.001                Minimal occupation difference for a transition
================  ==============  ===================  ========================================
