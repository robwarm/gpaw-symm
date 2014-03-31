from ase import *
from ase.structure import molecule
from gpaw import *
from gpaw.xc.hybridk import HybridXC
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

ecut = 25

N2 = molecule('N2')
N2.center(vacuum=2.0)

calc = GPAW(mode='pw', dtype=complex, xc='PBE', communicator=serial_comm)
N2.set_calculator(calc)
E_n2_pbe = N2.get_potential_energy()
E_n2_hf = E_n2_pbe + calc.get_xc_difference(HybridXC('EXX',
                                                     etotflag=True))
calc.diagonalize_full_hamiltonian(nbands=100)

rpa = RPACorrelation(calc, vcut='3D')
E_n2_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                          directions=[[0, 2/3.], [2, 1/3.]],
                                          gauss_legendre=8)

# -------------------------------------------------------------------------

N = molecule('N')
N.set_cell(N2.cell)

calc = GPAW(mode='pw', dtype=complex, xc='PBE', communicator=serial_comm)
N.set_calculator(calc)
E_n_pbe = N.get_potential_energy()
E_n_hf = E_n_pbe + calc.get_xc_difference(HybridXC('EXX',
                                                   etotflag=True))
calc.diagonalize_full_hamiltonian(nbands=100)

rpa = RPACorrelation(calc, vcut='3D')
E_n_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                         directions=[[0, 1.0]],
                                         gauss_legendre=8)
print 'Atomization energies:'
print 'PBE: ', E_n2_pbe - 2*E_n_pbe
print 'HF: ',  E_n2_hf - 2*E_n_hf
print 'HF+RPA: ', E_n2_hf - 2*E_n_hf + E_n2_rpa - 2*E_n_rpa, '(Not converged!)'

equal(E_n2_rpa - 2*E_n_rpa, -1.72, 0.02)
