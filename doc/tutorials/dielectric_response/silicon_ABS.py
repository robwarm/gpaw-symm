### Refer to G. Kresse, Phys. Rev. B 73, 045112 (2006)
### for comparison of macroscopic and microscopic dielectric constant 
### and absorption peaks. 

import os
import sys
import numpy as np

from ase.units import Bohr
from ase.lattice import bulk
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull
from gpaw.response.df import DF


if rank != 0:
    sys.stdout = devnull 

GS = 1
ABS = 1

if GS:

    # Ground state calculation
    a = 5.431 #10.16 * Bohr 
    atoms = bulk('Si', 'diamond', a=a)

    calc = GPAW(h=0.20,
            kpts=(12,12,12),
            xc='LDA',
            basis='dzp',
            txt='si_gs.txt',
            nbands=80,
            eigensolver='cg',
            occupations=FermiDirac(0.001),
            convergence={'bands':70})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('si.gpw','all')


if ABS:
            
    w = np.linspace(0, 24, 481)
    q = np.array([0.0, 0.00001, 0.])

    # getting macroscopic constant
    df = DF(calc='si.gpw', q=q, w=w, eta=0.0001, 
        hilbert_trans=False, txt='df_1.out',
        ecut=150, optical_limit=True)

    df.get_macroscopic_dielectric_constant()

    df.write('df_1.pckl')

    #getting absorption spectrum
    df = DF(calc='si.gpw', q=q, w=w, eta=0.1,
        ecut=150, optical_limit=True, txt='df_2.out')

    df.get_absorption_spectrum(filename='si_abs.dat')
    df.check_sum_rule()
    
    df.write('df_2.pckl')

