import numpy as np
from time import time, ctime
from datetime import timedelta
from ase.lattice import bulk
from ase.units import Hartree
from gpaw import GPAW, FermiDirac
from gpaw.response.gw import GW
from gpaw.mpi import serial_comm, world, rank
from gpaw.wavefunctions.pw import PW

starttime = time()

a = 3.567
atoms = bulk('C', 'diamond', a=a)

kpts = (2,2,2)

calc = GPAW(
            h=0.24,
            mode=PW(400),
            kpts=kpts,
            xc='LDA',
            txt='C_gs.txt',
            occupations=FermiDirac(0.001),
            usesymm=None,
            parallel={'band':1}
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=100)
calc.write('C_gs.gpw', 'all')

file='C_gs.gpw'

gw = GW(
        file=file,
        nbands=20,
        bands=np.array([3,4]),
        w=None,
        ecut=25.,
        eta=0.1,
        hilbert_trans=False
       )

gw.get_exact_exchange()
gw.get_QP_spectrum()

gap = (gw.QP_skn[0,0,1] - gw.QP_skn[0,0,0]) * Hartree

if not (np.abs(gap - 11.35) < 0.01):
    raise AssertionError("check your results!")

totaltime = round(time() - starttime)
print "GW test finished in %s " %(timedelta(seconds=totaltime))
