from math import cos, sin, pi

from ase import Atoms
from ase.optimize import QuasiNewton

from gpaw import GPAW

# Water molecule:
d = 0.9575
t = pi / 180 * 104.51

H2O = Atoms('H2O',
            positions=[(0, 0, 0),
                       (d, 0, 0),
                       (d * cos(t), d * sin(t), 0)])

H2O.center(vacuum=3.5)

calc = GPAW(h=0.2, txt='h2o.txt')

H2O.set_calculator(calc)

QuasiNewton(H2O).run(fmax=0.05)

calc.write('h2o.gpw')
