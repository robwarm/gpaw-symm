from ase import Atoms
from gpaw import GPAW
from gpaw.xc.noncollinear import NonCollinearFunctional, \
     NonCollinearLCAOEigensolver, NonCollinearMixer
from gpaw.xc import XC

a = 2.84
fe = Atoms('Fe2',
           scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
           magmoms=[2.3, 2.3],
           cell=(a, a, a),
           pbc=True)

k = 4
fe.calc = GPAW(txt='fec.txt', mode='lcao', basis='dz(dzp)', kpts=(k, k, k),
               convergence=dict(energy=1e-6))
e0 = fe.get_potential_energy()

fe.set_initial_magnetic_moments()
fe.set_initial_magnetic_moments([(2.3, 0, 0), (2.3, 0, 0)])
fe.calc = GPAW(txt='fenc.txt', mode='lcao', basis='dz(dzp)', kpts=(k, k, k),
               convergence=dict(energy=1e-6),
               usesymm=False,
               xc=NonCollinearFunctional(XC('LDA')),
               mixer=NonCollinearMixer(),
               eigensolver=NonCollinearLCAOEigensolver())
e = fe.get_potential_energy()

assert abs(e - e0) < 7e-5
