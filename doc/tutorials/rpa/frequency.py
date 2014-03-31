from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

ws = np.array([0.5*i for i in range(2000)])
ecut = 50

calc = GPAW('N2.gpw', communicator=serial_comm, txt=None)

rpa = RPACorrelation(calc, txt='frequency_equidistant.txt')

Es = rpa.get_E_q(ecut=ecut,
                 w=ws,
                 integrated=False,
                 q=[0,0,0],
                 direction=0)

f = paropen('frequency_equidistant.dat', 'w')
for w, E in zip(ws, Es):
    print >> f, w, E.real
f.close()
