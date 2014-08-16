from ase import *
from ase.lattice import bulk
from ase.dft import monkhorst_pack
from ase.parallel import paropen
from gpaw import *
from gpaw.wavefunctions.pw import PW
from gpaw.xc.exx import EXX

# Monkhorst-Pack grid shifted to be gamma centered
k = 8
kpts = monkhorst_pack([k, k, k])
kpts += [1. / (2 * k), 1. / ( 2 * k), 1. / (2 * k)]

cell = bulk('C', 'fcc', a=3.553).get_cell()
a = Atoms('C2', cell=cell, pbc=True,
          scaled_positions=((0, 0, 0), (0.25, 0.25, 0.25)))

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            convergence={'density': 1.e-6},
            kpts=kpts,
            txt='diamond_pbe.txt',
            )

a.set_calculator(calc)
E_pbe = a.get_potential_energy()

exx = EXX(calc, txt='diamond_exx.txt')
exx.calculate()
E_hf = exx.get_total_energy()

import numpy as np
E_C = np.loadtxt('PBE_HF_C.dat')

f = paropen('PBE_HF_diamond.dat', 'w')
print >> f, 'PBE: ', E_pbe / 2 - E_C[0]
print >> f, 'HF: ', E_hf / 2 - E_C[1]
f.close()

calc.diagonalize_full_hamiltonian()
calc.write('diamond.gpw', mode='all')
