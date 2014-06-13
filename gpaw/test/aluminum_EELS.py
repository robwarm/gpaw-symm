import numpy as np
import sys
import os
import time
from ase.units import Bohr
from ase.lattice import bulk
from gpaw import GPAW
from gpaw.eigensolvers.rmm_diis_old import RMM_DIIS
from gpaw.mixer import Mixer
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

assert size <= 4**3

# Ground state calculation

t1 = time.time()

a = 4.043
atoms = bulk('Al', 'fcc', a=a)
atoms.center()
calc = GPAW(h=0.2,
            eigensolver=RMM_DIIS(),
            mixer=Mixer(0.1,3),
            kpts=(4,4,4),
            parallel={'domain':1,
                      'band':1},
            idiotproof=False,  # allow uneven distribution of k-points
            xc='LDA')

atoms.set_calculator(calc)
atoms.get_potential_energy()
t2 = time.time()

# Excited state calculation
q = np.array([1/4.,0.,0.])
w = np.linspace(0, 24, 241)

df = DF(calc=calc, q=q, w=w, eta=0.2, ecut=(50,50,50))
df.get_EELS_spectrum(filename='EELS_Al')
df.check_sum_rule()
df.write('Al.pckl')

t3 = time.time()

print 'For ground  state calc, it took', (t2 - t1) / 60, 'minutes'
print 'For excited state calc, it took', (t3 - t2) / 60, 'minutes'

d = np.loadtxt('EELS_Al')
wpeak = 15.7 # eV
Nw = 157
if d[Nw, 1] > d[Nw-1, 1] and d[Nw, 2] > d[Nw+1, 2]:
    pass
else:
    raise ValueError('Plasmon peak not correct ! ')

if (np.abs(d[Nw, 1] - 28.8932274034) > 1e-5
    or np.abs(d[Nw, 2] -  25.9806674277) > 1e-5):
    print d[Nw, 1], d[Nw, 2]
    raise ValueError('Please check spectrum strength ! ')

