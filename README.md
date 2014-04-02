gpaw-symm
=========

Private project to add better symmetry usage in the GPAW code (https://wiki.fysik.dtu.dk/gpaw).
Starting from svn 11325.

It looks as if it is not compatible with 'fd' mode, because 'fd' needs grids divisible by 4, but
the gpaw initilization does not allow to set things in the order we need. It is pretty the problem,
that the symmetry part is a subclass of kpt_descriptor... not good for us.