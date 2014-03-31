from math import log
from ase import Atom, Atoms
from gpaw import GPAW, FermiDirac
from gpaw.test import equal

a = 4.0
h = 0.2
hydrogen = Atoms('H',
                 [(a / 2, a / 2, a / 2)],
                 cell=(a, a, a))

hydrogen.calc = GPAW(h=h, nbands=1, convergence={'energy': 1e-7})
e1 = hydrogen.get_potential_energy()
equal(e1, 0.526939, 0.001)

kT = 0.001
hydrogen.calc.set(occupations=FermiDirac(width=kT))
e2 = hydrogen.get_potential_energy()
equal(e1, e2 + log(2) * kT, 3.0e-7)
