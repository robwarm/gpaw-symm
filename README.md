gpaw-symm
=========

Private project to add better symmetry usage in the GPAW code (https://wiki.fysik.dtu.dk/gpaw).
Starting from svn 11325 (Update: rebased to svn 11542).

It looks as if it is not compatible with 'fd' mode, because 'fd' needs grids divisible by 4, but
the gpaw initilization does not allow to set things in the order we need. It is pretty much the problem,
that the symmetry part is a subclass of kpt_descriptor... not good for us.


IMPORTANT ISSUES:

- PW mode seems to work ..ish

- LCAO did not work for quartz, but perfectly for stishovite... weird

- reponse does not work properly, but I won't fix, because people are working on new version.
- the following tests will fail with 1 core: chi0.py
- the following tests will fail with 2+ cores: not tested yet
          