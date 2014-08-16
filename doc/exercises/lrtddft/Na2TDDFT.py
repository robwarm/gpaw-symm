from gpaw import GPAW
from ase import Atoms
from gpaw.lrtddft import LrTDDFT

molecule = Atoms('Na2', positions=((0.0, 0.0, 0.0), (3.12, 0.0, 0.0)))

molecule.center(vacuum=6.0)

calc = GPAW(xc='PBE')

molecule.set_calculator(calc)

molecule.get_potential_energy()

lr = LrTDDFT(calc, xc='LDA', istart=0, jend=10, nspins=2)
lr.write('Omega_Na2.gz')
