from ase import *
from ase.structure import molecule
from gpaw import *
from gpaw.wavefunctions.pw import PW
from gpaw.xc.fxc_correlation_energy import FXCCorrelation
from gpaw.test import equal
from gpaw.mpi import serial_comm, world, rank
from os import system

if world.size == 1:
    scalapack1 = None
    scalapack2 = None
elif world.size == 2:
    scalapack1 = (2, world.size // 2, 32)    
    scalapack2 = None
else:
    scalapack1 = (2, world.size // 2, 32)
    scalapack2 = (2, world.size // 4, 32)

# N2 --------------------------------------
N2 = molecule('N2')
N2.set_cell((2.5, 2.5, 3.5))
N2.center()
calc = GPAW(mode='pw',
            eigensolver='rmm-diis',
            dtype=complex,
            xc='LDA',
            nbands=16,
            basis='dzp',
            convergence={'density': 1.e-6})
N2.set_calculator(calc)
N2.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=80, scalapack=scalapack1)
calc.write('N2.gpw', mode='all')

calc = GPAW('N2.gpw', communicator=serial_comm, txt=None)
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='standard')
Ec_N2 = ralda.get_fxc_correlation_energy(ecut=50,
                                         directions=[[0, 2/3.], [2, 1/3.]])
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='solid')
Ec_N2_s = ralda.get_fxc_correlation_energy(ecut=50,
                                           directions=[[0, 2/3.], [2, 1/3.]])

# N ---------------------------------------
N = Atoms('N', [(0,0,0)])
N.set_cell((2.5, 2.5, 3.5))
N.center()
calc = GPAW(mode='pw',
            eigensolver='rmm-diis',
            dtype=complex,
            xc='LDA',
            basis='dzp',
            nbands=8,
            hund=True,
            convergence={'density': 1.e-6})
N.set_calculator(calc)
N.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=80, scalapack=scalapack2)
calc.write('N.gpw', mode='all')

calc = GPAW('N.gpw', communicator=serial_comm, txt=None)
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='standard',
                       )
Ec_N = ralda.get_fxc_correlation_energy(ecut=50,
                                        directions=[[0, 1.0]])
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='solid',
                       )
Ec_N_s = ralda.get_fxc_correlation_energy(ecut=50,
                                          directions=[[0, 1.0]])

if rank == 0:
   system('rm N2.gpw')
   system('rm N.gpw')

equal(Ec_N2, -6.1651, 0.001,)
equal(Ec_N2_s, -7.7042, 0.001,)
equal(Ec_N, -1.0567, 0.001)
equal(Ec_N_s, -2.0419, 0.001)
