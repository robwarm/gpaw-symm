from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np

name = 'sishovite'

alat = 4.2339442838332060
clat = 2.6932641034824649
a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([0.0, 1.0, 0.0])
a3 = clat*np.array([0.0, 0.0, 1.0])
cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si','O','O', 'O','O']

spos_ac = np.array([(0.0000000000000000,  0.0000000000000000, 0.0000000000000000),
                    (0.5000000000000000,  0.5000000000000000, 0.5000000000000000),
                    (0.3068662447268501,  0.3068662447268501, 0.0000000000000000),
                    (0.6931337552731499,  0.6931337552731499, 0.0000000000000000),
                    (0.1931337552731499,  0.8068662447268501, 0.5000000000000000),
                    (0.8068662447268501,  0.1931337552731499, 0.5000000000000000)])


atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc = True
              )

calc = GPAW(mode=PW(ecut=600,cell=cell_cv),
            xc='LDA',
            kpts=(3,3,3),
            nbands = 28,
            usefractrans = False,
            gpts = (24,24,16),
#            usesymm = False,
           )

atoms.set_calculator(calc)
energy = atoms.get_potential_energy()

# Get the accurate KS-band gap
homolumo = calc.occupations.get_homo_lumo(calc.wfs)
homo, lumo = homolumo
print "band gap ",(lumo-homo)*27.2

occs = calc.get_occupation_numbers()
print occs
