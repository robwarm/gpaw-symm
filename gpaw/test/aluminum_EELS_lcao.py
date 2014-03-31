import numpy as np
import sys
import os
import time
from ase.units import Bohr
from ase.lattice import bulk
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

assert size <= 4**3

# Ground state calculation

a = 4.043
atoms = bulk('Al', 'fcc', a=a)
atoms.center()
calc = GPAW(h=0.2,
            kpts=(4,4,4),
            nbands=13,
            mode='lcao',
            basis='dzp',
            xc='LDA')

atoms.set_calculator(calc)
t1 = time.time()
atoms.get_potential_energy()
t2 = time.time()
calc.write('Al.gpw','all')

t3 = time.time()

# Excited state calculation
q = np.array([1/4.,0.,0.])
w = np.linspace(0, 24, 241)
    
df = DF(calc='Al.gpw', q=q, w=w, eta=0.2, ecut=50)
#df.write('Al.pckl')
df.get_EELS_spectrum(filename='EELS_Al_lcao')
df.check_sum_rule()
    
t4 = time.time()

print 'For ground  state calc, it took', (t2 - t1) / 60, 'minutes'
print 'For writing gpw, it took', (t3 - t2) / 60, 'minutes'
print 'For excited state calc, it took', (t4 - t3) / 60, 'minutes'

d = np.loadtxt('EELS_Al_lcao')
wpeak = 16.9 # eV
Nw = 169
if d[Nw, 1] > d[Nw-1, 1] and d[Nw, 2] > d[Nw+1, 2]:
    pass
else:
    raise ValueError('Plasmon peak not correct ! ')

if (np.abs(d[Nw, 1] - 19.7274875955) > 1e-3
    or np.abs(d[Nw, 2] -  18.9147047194) > 1e-3):
    print d[Nw, 1], d[Nw, 2]
    raise ValueError('Please check spectrum strength ! ')
                                                              


