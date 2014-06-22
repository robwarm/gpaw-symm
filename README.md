gpaw-symm
=========

Private project to add better symmetry usage in the GPAW code (https://wiki.fysik.dtu.dk/gpaw).
Starting from svn 11325 (Update: rebased to svn 11511).

It looks as if it is not compatible with 'fd' mode, because 'fd' needs grids divisible by 4, but
the gpaw initilization does not allow to set things in the order we need. It is pretty the problem,
that the symmetry part is a subclass of kpt_descriptor... not good for us.

Current version seems to work for orthorhombic and related cells (e.g. stishovite), but not with trigonal (e.g. quartz).
Funny that is.

IMPORTANT ISSUES:

- after lastest update get funy numpy warnings, which don't make sense:
/home/rbw/programs/gpaw/gpaw/symmetry.py:172: RuntimeWarning: divide by zero encountered in divide
  invft = np.where( np.abs(ft) > 0.01, 1./ft, 0.)
/home/rbw/programs/gpaw/gpaw/symmetry.py:176: RuntimeWarning: divide by zero encountered in divide
  ft = np.where( np.abs(invft_int) > 1e-4, 1./invft_int, 0.)


- rpa does not work, but I won't fix, because people are working on new version.
  The following tests will fail:
            rpa_energy_Si.py
            chi0.py
            