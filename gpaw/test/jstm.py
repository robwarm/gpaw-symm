import cPickle as pickle
import numpy as np
from ase import Atoms
from gpaw import GPAW, setup_paths, FermiDirac
from gpaw.transport.jstm import STM, dump_hs, dump_lead_hs
from gpaw.atom.basis import BasisMaker
from gpaw.mpi import world
from gpaw.test import equal

if world.rank == 0:
    basis = BasisMaker('H', 'sz').generate(1, 0)
    basis.write_xml()
world.barrier()
if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

# GPAW calculations
a = 0.75 # Bond length
cell = np.diag([5, 5, 12 * a])

atoms = Atoms('H12', pbc=(1, 1, 1), cell=cell)
atoms.positions[:, 2] = [i * a for i in range(12)]

calc = GPAW(h=0.2,
            occupations=FermiDirac(width=0.1),
            mode='lcao',
            basis='sz',
            usesymm=False)

# Lead calculation
lead = atoms.copy()
del lead[4:]
cell_lead = cell.copy()
cell_lead[2, 2] = a * len(lead)
lead.set_cell(cell_lead)
calc.set(kpts=(1, 1, 14))
lead.set_calculator(calc)
e1 = lead.get_potential_energy()
niter1 = calc.get_number_of_iterations()
calc.write('lead.gpw')
dump_lead_hs(calc, 'lead')

# Tip calculation
tip = atoms.copy()
tip.positions[:] += (atoms.cell / 2.0)[0, :] + (atoms.cell / 2.0)[1, :]
del tip[:4]
tip.translate([0, 0, 10])
tip.cell[2, 2] += 10
calc.set(kpts=(1, 1, 1))
tip.set_calculator(calc)
e2 = tip.get_potential_energy()
niter2 = calc.get_number_of_iterations()
calc.write('tip.gpw')
dump_hs(calc, 'tip', region='tip', cvl=2)

# Surface calculation
srf = atoms.copy()
del srf[8:]
srf.cell[2, 2] += 10
srf.set_calculator(calc)
e3 = srf.get_potential_energy()
niter3 = calc.get_number_of_iterations()
calc.write('srf.gpw', region='surface', cvl=2)
dump_hs(calc, 'srf', region='surface', cvl=2)

#STM simulation
tip = GPAW('tip')
srf = GPAW('srf')

h0, s0 = pickle.load(open('lead_hs.pckl'))
h1, s1 = pickle.load(open('tip_hs.pckl'))
h2, s2 = pickle.load(open('srf_hs.pckl'))

assert abs(h0.imag).max() < 1e-14
assert abs(s0.imag).max() < 1e-14
h0 = h0.real
s0 = h0.real

stm = STM(tip, srf,
          hs10=(h0[0], s0[0]),
          hs1=(h1[0], s1[0]),
          hs2=(h2[0], s2[0]),
          hs20=(h0[0], s0[0]),
          align_bf=0)

stm.set(dmin=5)
stm.initialize()

stm.scan()
stm.linescan()

if 0:
    stm.plot(repeat=[3, 3])

energy_tolerance = 0.0003
niter_tolerance = 0
equal(e1, -5.29095, energy_tolerance)
equal(e2, -12.724, energy_tolerance)
equal(e3, -12.7236, energy_tolerance)
