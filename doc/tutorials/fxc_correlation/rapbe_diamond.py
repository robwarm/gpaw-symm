from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation

fxc = FXCCorrelation('diamond.gpw', xc='rAPBE', txt='rapbe_diamond.txt')    
E_i = fxc.calculate(ecut=400)

f = paropen('rapbe_diamond.dat', 'w')
for ecut, E in zip(fxc.ecut_i, E_i):
    print >> f, ecut * Hartree, E
f.close()

rpa = RPACorrelation('diamond.gpw', txt='rpa_diamond.txt')    
E_i = rpa.calculate(ecut=400)

f = paropen('rpa_diamond.dat', 'w')
for ecut, E in zip(rpa.ecut_i, E_i):
    print >> f, ecut * Hartree, E
f.close()
