from ase import Atoms
from gpaw import GPAW
from gpaw import PW
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from gpaw.symmetry import Symmetry
#from gpaw.symmetry2 import Symmetry2

name = 'quartz'

alat = 5.032090
clat = alat * 1.09685337425

a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([-0.5, np.sqrt(3)/2., 0.0])
a3 = clat*np.array([0.0, 0.0, 1.])

cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si', 'Si','O', 'O', 'O','O', 'O','O']

#spos_ac = np.array([(0.4789444,  0.0000000,  0.3333333), 
                    #(0.0000000,  0.4789444,  0.6666667), 
                    #(0.5210556,  0.5210556,  0.0000000),
                    #(0.4152831,  0.2510908,  0.2014540), 
                    #(0.7489090,  0.1641924,  0.5347874), 
                    #(0.8358074,  0.5847168,  0.8681209), 
                    #(0.2510908,  0.4152831,  0.7985463), 
                    #(0.1641923,  0.7489090,  0.4652128), 
                    #(0.5847168,  0.8358075,  0.1318794)]) 

spos_ac = np.array([(0.47894440,  0.00000000, 0.33333333), 
                    (0.00000000,  0.47894440, 0.66666667), 
                    (0.52105560,  0.52105560, 0.00000000),
                    (0.41528322,  0.25109082, 0.20145397), 
                    (0.74890918,  0.16419240, 0.53478730), 
                    (0.83580760,  0.58471678, 0.86812063), 
                    (0.25109082,  0.41528322, 0.79854603), 
                    (0.16419240,  0.74890918, 0.46521270), 
                    (0.58471678,  0.83580760, 0.13187937)]) 


atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc = True
              )

# first GS calc
calc = GPAW( #mode=PW(ecut=600,cell=cell_cv),
            #h = 0.20,
            mode= 'lcao',
            xc='LDA',
            kpts=(3,3,3),
#            txt='gs.out',
#            nbands = 42,
            lft = False,
#            eigensolver = 'rmm-diis',
#            gpts = (25,25,33),
            gpts = (20,20,30),
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
