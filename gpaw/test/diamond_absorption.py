import numpy as np
import sys
import time

from ase.units import Bohr
from ase.lattice import bulk
from gpaw import GPAW, FermiDirac
from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.mixer import Mixer
from gpaw.atom.basis import BasisMaker
from gpaw.response.df import DF
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull


if rank != 0:
  sys.stdout = devnull 

# GS Calculation One
a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(h=0.2,
            eigensolver=RMM_DIIS(),
            mixer=Mixer(0.1,3),
            kpts=(4,4,4),
            occupations=FermiDirac(0.001))

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C.gpw','all')

# Macroscopic dielectric constant calculation
q = np.array([0.0, 0.00001, 0.])
w = np.linspace(0, 24., 241)

df = DF(calc='C.gpw', q=q, w=(0.,), eta=0.001,
        ecut=50, hilbert_trans=False, optical_limit=True)
eM1, eM2 = df.get_macroscopic_dielectric_constant()

eM1_ = 6.15176021 #6.15185095143 for dont use time reversal symmetry
eM2_ = 6.04805705 #6.04815084635

if (np.abs(eM1 - eM1_) > 1e-5 or
    np.abs(eM2 - eM2_) > 1e-5):
    print eM1, eM2
    raise ValueError('Macroscopic dielectric constant not correct ! ')


# Absorption spectrum calculation
del df
df = DF(calc='C.gpw', q=q, w=w, eta=0.25,
        ecut=50, optical_limit=True, txt='C_df.out')
df.get_absorption_spectrum()
df.check_sum_rule()
df.write('C_df.pckl')
