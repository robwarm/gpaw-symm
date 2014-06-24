from ase import Atoms
from gpaw import GPAW
import numpy as np

name = 'quartz'

alat = 5.032090
clat = alat * 1.09685337425
a1 = alat*np.array([1.0, 0.0, 0.0])
a2 = alat*np.array([-0.5, np.sqrt(3)/2., 0.0])
a3 = clat*np.array([0.0, 0.0, 1.])
cell_cv = np.array([a1,a2,a3])

symbols = ['Si', 'Si', 'Si','O', 'O', 'O','O', 'O','O']

#spos_ac = np.array([(0.47894440,  0.00000000, 0.33333333), 
                    #(0.00000000,  0.47894440, 0.66666667), 
                    #(0.52105560,  0.52105560, 0.00000000),
                    #(0.41528322,  0.25109082, 0.20145397), 
                    #(0.74890918,  0.16419240, 0.53478730), 
                    #(0.83580760,  0.58471678, 0.86812063), 
                    #(0.25109082,  0.41528322, 0.79854603), 
                    #(0.16419240,  0.74890918, 0.46521270), 
                    #(0.58471678,  0.83580760, 0.13187937)]) 

spos_ac = np.array([( 0.4778762817077312,  0.0000000000000000,  0.3333333333333333), 
                    ( 0.0000000000000000,  0.4778762817077312,  0.6666666666666666), 
                    ( 0.5221237182922689,  0.5221237182922689,  0.0000000000000000),
                    ( 0.4153075513810672,  0.2531339617721680,  0.2029892900232357), 
                    ( 0.7468660382278319,  0.1621735896088991,  0.5363226233565690), 
                    ( 0.8378264103911008,  0.5846924486189328,  0.8696559566899023), 
                    ( 0.2531339617721680,  0.4153075513810672,  0.7970107099767644), 
                    ( 0.1621735896088991,  0.7468660382278319,  0.4636773766434310), 
                    ( 0.5846924486189328,  0.8378264103911008,  0.1303440433100977)]) 


atoms = Atoms(symbols=symbols,
              scaled_positions=spos_ac,
              cell=cell_cv,
              pbc = True
              )

calc = GPAW(mode= 'lcao',
            xc='LDA',
            kpts=(3,3,3),
            #usefractrans = False,
            gpts = (20,20,30),
            usesymm = False,
           )

atoms.set_calculator(calc)
energy = atoms.get_potential_energy()

# Get the accurate KS-band gap
homolumo = calc.occupations.get_homo_lumo(calc.wfs)
homo, lumo = homolumo
print "band gap ",(lumo-homo)*27.2

occs = calc.get_occupation_numbers()
print occs
