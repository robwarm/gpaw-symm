from ase.lattice import bulk
from gpaw import GPAW
from gpaw.wavefunctions.pw import PW

pwcutoff = 600.0 # Plane wave cutoff

k = 8 # NxNxN k-point sampling, gamma-centred grid

alat = 5.421 # Si lattice constant


# Do the bulk calculation


bulk_crystal = bulk('Si', 'diamond', a=alat)   
bulk_calc = GPAW(
            mode = PW(pwcutoff),
            kpts={'size': (k, k, k), 'gamma': True},
            xc='PBE',
            txt='si.pbe_output.txt',
            parallel={'band':1}
            )

bulk_crystal.set_calculator(bulk_calc)
e0_bulk_pbe = bulk_crystal.get_potential_energy()
