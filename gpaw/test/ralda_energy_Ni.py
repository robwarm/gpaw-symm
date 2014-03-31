from ase import *
from ase.lattice import bulk
from ase.dft.kpoints import monkhorst_pack
from gpaw import *
from gpaw.test import equal
from gpaw.xc.fxc_correlation_energy import FXCCorrelation
from gpaw.mpi import serial_comm, world

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

ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='solid')
E_solid = ralda.get_fxc_correlation_energy(ecut=50,
                                           gauss_legendre=8,
                                           skip_gamma=True,
                                           kcommsize=world.size)

ralda = FXCCorrelation(calc,
                       xc='rALDA',
                       method='standard',
                       unit_cells=[2,1,1],
                       )
E_standard = ralda.get_fxc_correlation_energy(ecut=50,
                                              gauss_legendre=8,
                                              skip_gamma=True,
                                              kcommsize=world.size)

equal(E_solid, -7.391, 0.001)
equal(E_standard, -7.501, 0.001)
