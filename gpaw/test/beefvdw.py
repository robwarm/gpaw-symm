from ase import *
from ase.dft.bee import BEEF_Ensemble
from gpaw import GPAW
from gpaw.test import equal
import numpy as np

xc = 'BEEF-vdW'
d = 0.75
tol1 = 1.e-10
tol2 = 1.e-2
tol3 = 1.e-1

# H2 molecule
h2 = Atoms('H2',[[0.,0.,0.],[0.,0.,d]])
h2.center(vacuum=2.)
cell = h2.get_cell()
calc = GPAW(xc=xc)
h2.set_calculator(calc)
e_h2 = h2.get_potential_energy()
f = h2.get_forces()
ens = BEEF_Ensemble(calc)
de_h2 = ens.get_ensemble_energies()
del h2, calc, ens

# H atom
h = Atoms('H')
h.set_cell(cell)
h.center()
calc = GPAW(xc=xc, spinpol=True)
h.set_calculator(calc)
e_h = h.get_potential_energy()
ens = BEEF_Ensemble(calc)
de_h = ens.get_ensemble_energies()

# forces
f0 = f[0].sum()
f1 = f[1].sum()
equal(f0, -f1, tol1)
equal(f0, 1.044, tol2)

# binding energy
E_bind = 2*e_h - e_h2
dE_bind = 2*de_h[:] - de_h2[:]
dE_bind = np.std(dE_bind)
equal(E_bind, 5.126, tol2)
equal(dE_bind, 0.2, tol3)
