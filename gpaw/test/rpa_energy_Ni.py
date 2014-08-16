from ase import *
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import *
from gpaw.mpi import serial_comm, world
from gpaw.test import equal
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation
import numpy as np

a0  = 5.43
Ni = bulk('Ni', 'fcc')
Ni.set_initial_magnetic_moments([0.7])

kpts = monkhorst_pack((3,3,3))

calc = GPAW(mode='pw',
            kpts=kpts,
            occupations=FermiDirac(0.001),
            setups={'Ni': '10'}, 
            communicator=serial_comm)
Ni.set_calculator(calc)
E = Ni.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=50)

rpa = RPACorrelation(calc, nfrequencies=8, skip_gamma=True)
E_rpa = rpa.calculate(ecut=[50])

fxc = FXCCorrelation(calc, nlambda=16, nfrequencies=8, skip_gamma=True)
E_fxc = fxc.calculate(ecut=[50])

equal(E_rpa, -7.826, 0.01)
equal(E_fxc, -7.826, 0.01)
