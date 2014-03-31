import numpy as np

from ase.structure import molecule
from gpaw import GPAW

unitcell = np.array([6.5, 6.6, 9.])
gridrefinement = 2

for formula in ('Na', 'Cl', 'NaCl',):

    calc = GPAW(xc='PBE',
                nbands=-5,
                h=0.18,
                convergence={'eigenstates':1E-8},
                txt=formula + '.txt')

    if formula == 'Cl':
        calc.set(hund=True)

    sys = molecule(formula, cell=unitcell, calculator=calc)
    sys.center()
    sys.get_potential_energy()

    # Get densities
    nt = calc.get_pseudo_density()
    n = calc.get_all_electron_density(gridrefinement=gridrefinement)

    # Get integrated values
    dv = sys.get_volume() / calc.get_number_of_grid_points().prod()
    It = nt.sum() * dv
    I = n.sum() * dv / gridrefinement**3

    print '%-4s %4.2f %5.2f' % (formula, It, I)
