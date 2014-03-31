from ase import *
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

# H2 --------------------------------------
H2 = Atoms('H2', [(0,0,0),(0,0,0.7413)])
H2.set_pbc(True)
H2.set_cell((2., 2., 3.))
H2.center()
calc = GPAW(mode=PW(210),
            eigensolver='rmm-diis',
            dtype=complex,
            #spinpol=True,
            xc='LDA',
            basis='dzp',
            nbands=8,
            convergence={'density': 1.e-6})
H2.set_calculator(calc)
H2.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=80, scalapack=scalapack1)
calc.write('H2.gpw', mode='all')
calc = GPAW('H2.gpw', communicator=serial_comm, txt=None)
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       )
Ec_H2 = ralda.get_fxc_correlation_energy(ecut=200,
                                         directions=[[0, 2/3.], [2, 1/3.]])

# H ---------------------------------------
H = Atoms('H', [(0,0,0)])
H.set_pbc(True)
H.set_cell((2., 2., 3.))
H.center()
calc = GPAW(mode=PW(210),
            eigensolver='rmm-diis',
            dtype=complex,
            xc='LDA',
            basis='dzp',
            nbands=4,
            hund=True,
            convergence={'density': 1.e-6})
H.set_calculator(calc)
H.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=80, scalapack=scalapack2)
calc.write('H.gpw', mode='all')
calc = GPAW('H.gpw', communicator=serial_comm, txt=None)
ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       )
Ec_H = ralda.get_fxc_correlation_energy(ecut=200,
                                        directions=[[0, 1.0]])

if rank == 0:
    system('rm H2.gpw')
    system('rm H.gpw')

equal(Ec_H2, -0.8411, 0.001)
equal(Ec_H, 0.003248, 0.00001)
