from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

ecut = 200

ds = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0, 6.0, 10.0]

for d in ds:
    calc = GPAW('gs_%s.gpw' % d, communicator=serial_comm, txt=None)
    rpa = RPACorrelation(calc, txt='rpa_%s_%s.txt' % (ecut, d))
    E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut,
                                           frequency_cut=800,
                                           frequency_scale=2.5,
                                           kcommsize=128,
                                           skip_gamma=True,
                                           #directions=[[0, 2/3.], [2, 1/3.]],
                                           restart='restart_%s_%s.txt' % (ecut, d))

    f = paropen('rpa_%s.dat' % ecut, 'a')
    print >> f, d, E_rpa
    f.close()
