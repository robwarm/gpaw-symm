from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.xc.exx import EXX
from gpaw.wavefunctions.pw import PW

# CO ------------------------------------------

CO = Atoms('CO', [(0, 0, 0), (0, 0, 1.1283)])
CO.set_pbc(True)
CO.center(vacuum=3.0)
calc = GPAW(mode=PW(600),
            dtype=complex,
            xc='PBE',
            txt='CO_pbe.txt',
            convergence={'density': 1.e-6})

CO.set_calculator(calc)
E0_pbe= CO.get_potential_energy()

exx = EXX(calc, txt='CO_exx.txt')
exx.calculate()
E0_hf = exx.get_total_energy()

calc.diagonalize_full_hamiltonian()
calc.write('CO.gpw', mode='all')

# C -------------------------------------------

C = Atoms('C')
C.set_pbc(True)
C.set_cell(CO.cell)
C.center()
calc = GPAW(mode=PW(600),
            dtype=complex,
            xc='PBE',
            mixer=MixerSum(beta=0.1, nmaxold=5, weight=50.0),
            hund=True,
            txt='C.txt',
            convergence={'density': 1.e-6})

C.set_calculator(calc)
E1_pbe = C.get_potential_energy()

exx = EXX(calc, txt='C_exx.txt')
exx.calculate()
E1_hf = exx.get_total_energy()

f = paropen('PBE_HF_C.dat', 'w')
print >> f, E1_pbe, E1_hf
f.close()

calc.diagonalize_full_hamiltonian()
calc.write('C.gpw', mode='all')

# O -------------------------------------------

O = Atoms('O')
O.set_pbc(True)
O.set_cell(CO.cell)
O.center()
calc = GPAW(mode=PW(600),
            dtype=complex,
            xc='PBE',
            mixer=MixerSum(beta=0.1, nmaxold=5, weight=50.0),
            hund=True,
            txt='O.txt',
            convergence={'density': 1.e-6})

O.set_calculator(calc)
E2_pbe = O.get_potential_energy()

exx = EXX(calc, txt='O_exx.txt')
exx.calculate()
E2_hf = exx.get_total_energy()

calc.diagonalize_full_hamiltonian()
calc.write('O.gpw', mode='all')

f = paropen('PBE_HF_CO.dat', 'w')
print >> f, 'PBE: ', E0_pbe - E1_pbe - E2_pbe
print >> f, 'HF: ', E0_hf - E1_hf - E2_hf
f.close()
