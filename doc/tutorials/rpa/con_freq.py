from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation

f = paropen('con_freq.dat', 'w')
for N in [4, 6, 8, 12, 16, 24, 32]:
    rpa = RPACorrelation('N2.gpw', txt='rpa_N2_frequencies.txt', nfrequencies=N)
    E = rpa.calculate(ecut=[50])
    print >> f, N, E[0]
    if N == 16:
        f16 = paropen('frequency_gauss16.dat', 'w')
        for w, e in zip(rpa.omega_w, rpa.E_w):
            print >> f16, w * Hartree, e
        f16.close()
f.close()
