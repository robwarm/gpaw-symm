from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation
from gpaw.mpi import world

fxc0 = FXCCorrelation('CO.gpw', xc='rAPBE', txt='rapbe_CO.txt', wcomm=world.size)    
fxc1 = FXCCorrelation('C.gpw', xc='rAPBE', txt='rapbe_C.txt', wcomm=world.size)    
fxc2 = FXCCorrelation('O.gpw', xc='rAPBE', txt='rapbe_O.txt', wcomm=world.size)

E0_i = fxc0.calculate(ecut=400)
E1_i = fxc1.calculate(ecut=400)
E2_i = fxc2.calculate(ecut=400)

f = paropen('rapbe_CO.dat', 'w')
for ecut, E0, E1, E2 in zip(fxc0.ecut_i, E0_i, E1_i, E2_i):
    print >> f, ecut * Hartree, E0 - E1 - E2, E1
f.close()

rpa0 = RPACorrelation('CO.gpw', txt='rpa_CO.txt', wcomm=world.size)    
rpa1 = RPACorrelation('C.gpw', txt='rpa_C.txt', wcomm=world.size)    
rpa2 = RPACorrelation('O.gpw', txt='rpa_O.txt', wcomm=world.size)

E0_i = rpa0.calculate(ecut=400)
E1_i = rpa1.calculate(ecut=400)
E2_i = rpa2.calculate(ecut=400)

f = paropen('rpa_CO.dat', 'w')
for ecut, E0, E1, E2 in zip(rpa0.ecut_i, E0_i, E1_i, E2_i):
    print >> f, ecut * Hartree, E0 - E1 - E2, E1
f.close()
