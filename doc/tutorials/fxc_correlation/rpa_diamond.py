from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

calc = GPAW('gs_12_lda.gpw', communicator=serial_comm, txt=None)
rpa = RPACorrelation(calc, txt='rpa_12_lda.txt')

#for ecut in [100,150,200,250,300,350,400]:
for ecut in [400, 450, 500, 550]:
    E = rpa.get_rpa_correlation_energy(ecut=ecut,
                                       directions=[[0, 1.0]],
                                       kcommsize=32)

    f = paropen('rpa_12_lda.dat', 'a')
    print >> f, ecut, E
    f.close()
