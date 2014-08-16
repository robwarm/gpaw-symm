from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.wavefunctions.pw import PW
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation

# LDA --------------------------------------

H = Atoms('H', [(0, 0, 0)])
H.set_pbc(True)
H.center(vacuum=2.0)
calc = GPAW(mode=PW(300),
            hund=True,
            dtype=complex,
            xc='LDA')

H.set_calculator(calc)
E_lda = H.get_potential_energy()
E_c_lda = -calc.get_xc_difference('LDA_X')

print 'LDA correlation: ', E_c_lda, 'eV'
print

calc.diagonalize_full_hamiltonian()
calc.write('H_lda.gpw', mode='all')

rpa = RPACorrelation('H_lda.gpw')
rpa.calculate(ecut=300)

fxc = FXCCorrelation('H_lda.gpw', xc='rALDA')
fxc.calculate(ecut=300)

# PBE --------------------------------------

H = Atoms('H', [(0, 0, 0)])
H.set_pbc(True)
H.center(vacuum=2.0)
calc = GPAW(mode=PW(300),
            hund=True,
            dtype=complex,
            xc='PBE')

H.set_calculator(calc)
E_pbe = H.get_potential_energy()
E_c_pbe = -calc.get_xc_difference('GGA_X_PBE')

print 'PBE correlation: ', E_c_pbe, 'eV'
print

calc.diagonalize_full_hamiltonian()
calc.write('H_pbe.gpw', mode='all')

rpa = RPACorrelation('H_pbe.gpw')
rpa.calculate(ecut=300)

fxc = FXCCorrelation('H_pbe.gpw', xc='rAPBE')
fxc.calculate(ecut=300)
