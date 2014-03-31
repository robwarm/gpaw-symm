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

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

kpts = (2,2,2)

calc = GPAW(
            h=0.24,
            mode=PW(100),
            kpts=kpts,
            xc='LDA',
            txt='Si_gs.txt',
            occupations=FermiDirac(0.001),
            usesymm=None,
            parallel={'band':1}
           )

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_gs.gpw','all')

file='Si_gs.gpw'

gw = GW(
        file=file,
        nbands=8,
        bands=np.array([3,4]),
        w=np.array([10., 30., 0.05]),
        ecut=25.,
        eta=0.1,
        hilbert_trans=False,
       )

gw.get_QP_spectrum()

gap = (gw.QP_skn[0,0,1] - gw.QP_skn[0,0,0]) * Hartree

if not (np.abs(gap - 3.48) < 0.01):
    raise AssertionError("check your results!")

totaltime = round(time() - starttime)
print "GW test finished in %s " %(timedelta(seconds=totaltime))
