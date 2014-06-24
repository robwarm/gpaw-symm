from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from gpaw.symmetry import Symmetry
#from gpaw.symmetry2 import Symmetry2

name = 'sishovite'

alat = 4.2339442838332060
clat = 2.6932641034824649

a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([0.0, 1.0, 0.0])
a3 = clat*np.array([0.0, 0.0, 1.0])

cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si','O','O', 'O','O']

spos_ac = np.array([(0.0000000000000000,  0.0000000000000000, 0.0000000000000000),
                    (0.5000000000000000,  0.5000000000000000, 0.5000000000000000),
                    (0.3068662447268501,  0.3068662447268501, 0.0000000000000000),
                    (0.6931337552731499,  0.6931337552731499, 0.0000000000000000),
                    (0.1931337552731499,  0.8068662447268501, 0.5000000000000000),
                    (0.8068662447268501,  0.1931337552731499, 0.5000000000000000)])


atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc = True
              )

# first GS calc
calc = GPAW( #mode=PW(cell=cell_cv),
            #h = 0.20,
            mode= 'lcao',
            xc='LDA',
            kpts=(3,3,3),
#            txt='gs.out',
#            nbands = 28,
            usefractrans = True,
#            eigensolver = 'rmm-diis',
            gpts = (20,20,12),
#            gpts = (20,20,30),
#            usesymm = False,
#            convergence={'bands':50}
#            occupations=FermiDirac(0.05)
            )

atoms.set_calculator(calc)
# Calculate the ground state
energy = atoms.get_potential_energy()
# Save the ground state
#calc.write('sio2_gs.gpw', 'all')

# Get the accurate KS-band gap
homolumo = calc.occupations.get_homo_lumo(calc.wfs)
homo, lumo = homolumo
print "band gap ",(lumo-homo)*27.2

occs = calc.get_occupation_numbers()
print occs
