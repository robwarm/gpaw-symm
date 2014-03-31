from ase import *
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.test import equal
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

a0 = 5.43
cell = bulk('Si', 'fcc', a=a0).get_cell()
Si = Atoms('Si2', cell=cell, pbc=True,
           scaled_positions=((0,0,0), (0.25,0.25,0.25)))

kpts = monkhorst_pack((2,2,2))
kpts += np.array([1/4., 1/4., 1/4.])

calc = GPAW(mode='pw',
            kpts=kpts,
            occupations=FermiDirac(0.001),
            communicator=serial_comm)
Si.set_calculator(calc)
E = Si.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=50)

ecut = 50
rpa = RPACorrelation(calc, qsym=False)
E_rpa_noqsym = rpa.get_rpa_correlation_energy(ecut=ecut,
                                              directions=[[0, 1.0]],
                                              gauss_legendre=8)

rpa = RPACorrelation(calc, qsym=True)
E_rpa_qsym = rpa.get_rpa_correlation_energy(ecut=ecut,
                                            directions=[[0, 1.0]],
                                            gauss_legendre=8)

equal(E_rpa_qsym, E_rpa_noqsym, 0.001)
equal(E_rpa_qsym, -12.61, 0.01)
