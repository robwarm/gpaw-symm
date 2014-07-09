from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np
from gpaw.test import equal

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

## with fractional translation
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3,3,3),
            nbands = 28,
            usefractrans = True,
            gpts = (18,18,12),
           )

atoms.set_calculator(calc)
energy_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 6)
assert(len(calc.wfs.kd.symmetry.op_scc) == 16)

## without fractional translations
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3,3,3),
            nbands = 28,
            usefractrans = False,
            gpts = (18,18,12),
           )
atoms.set_calculator(calc)
energy_no_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 8)
assert(len(calc.wfs.kd.symmetry.op_scc) == 8)

equal(energy_fractrans, energy_no_fractrans, 1e-7)