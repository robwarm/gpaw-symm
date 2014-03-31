from ase import *
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import *
from gpaw.mpi import serial_comm, world
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
from gpaw.xc.fxc_correlation_energy import FXCCorrelation
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

rpa = RPACorrelation(calc)
E_rpa = rpa.get_rpa_correlation_energy(ecut=50,
                                       skip_gamma=True,
                                       gauss_legendre=8)

fxc = FXCCorrelation(calc, xc='RPA')
E_fxc = fxc.get_fxc_correlation_energy(ecut=50,
                                       skip_gamma=True,
                                       gauss_legendre=8)

equal(E_rpa, -7.826, 0.01)
equal(E_fxc, -7.827, 0.01)
