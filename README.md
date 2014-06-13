gpaw-symm
=========

Private project to add better symmetry usage in the GPAW code (https://wiki.fysik.dtu.dk/gpaw).
Starting from svn 11325 (Update: rebased to svn 11511).

It looks as if it is not compatible with 'fd' mode, because 'fd' needs grids divisible by 4, but
the gpaw initilization does not allow to set things in the order we need. It is pretty the problem,
that the symmetry part is a subclass of kpt_descriptor... not good for us.


IMPORTANT ISSUES:

- rpa does not work, but I won't fix, because people are working on new version.
  The following tests will fail:
            rpa_energy_Si.py
            chi0.py
            