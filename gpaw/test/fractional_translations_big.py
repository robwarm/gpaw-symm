from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np
from gpaw.test import equal

name = 'cristobalite'

alat = 5.0833674013366767
clat = 7.0984737604292851
a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([0.0, 1.0, 0.0])
a3 = clat*np.array([0.0, 0.0, 1.0])
cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si', 'Si', 'Si', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

spos_ac = np.array([(0.2939117942733036,  0.2939117942733036,  0.0000000000000000),
                    (0.7060882057266963,  0.7060882057266963,  0.5000000000000000),
                    (0.2060882057266965,  0.7939117942733037,  0.2500000000000000),
                    (0.7939117942733037,  0.2060882057266965,  0.7500000000000000),
                    (0.2412655541827894,  0.0931313916609316,  0.1739217372147955),
                    (0.7587344458172105,  0.9068686083390685,  0.6739217372147954),
                    (0.4068686083390686,  0.7412655541827895,  0.4239217372147952),
                    (0.5931313916609315,  0.2587344458172107,  0.9239217372147954),
                    (0.0931313916609316,  0.2412655541827894,  0.8260782627852046),
                    (0.9068686083390685,  0.7587344458172105,  0.3260782627852048),
                    (0.2587344458172107,  0.5931313916609315,  0.0760782627852045),
                    (0.7412655541827895,  0.4068686083390686,  0.5760782627852046)])


atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc = True
              )
## with fractional translations
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3,3,2),
            nbands = 40,
            usefractrans = True,
            gpts = (24,24,32),
           )
atoms.set_calculator(calc)
energy_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 3)
assert(len(calc.wfs.kd.symmetry.op_scc) == 8)

## without fractional translations
calc = GPAW(mode=PW(),
            xc='LDA',
            kpts=(3,3,2),
            nbands = 40,
            usefractrans = False,
            gpts = (24,24,32),
           )
atoms.set_calculator(calc)
energy_no_fractrans = atoms.get_potential_energy()

assert(len(calc.wfs.kd.ibzk_kc) == 6)
assert(len(calc.wfs.kd.symmetry.op_scc) == 2)

equal(energy_fractrans, energy_no_fractrans, 1e-7)