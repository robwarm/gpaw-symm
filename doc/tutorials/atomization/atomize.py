from ase import Atoms, Atom
from gpaw import GPAW

a = 4.  # Size of unit cell (Angstrom)
c = a / 2
# Hydrogen atom:
atom = Atoms('H',
             positions=[(c, c, c)],
             magmoms=[1],
             cell=(a, a, a))

# gpaw calculator:
calc = GPAW(h=0.18, nbands=1, xc='PBE', txt='H.out')
atom.set_calculator(calc)

e1 = atom.get_potential_energy()
calc.write('H.gpw')

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = Atoms('H2',
                 positions=([c - d / 2, c, c],
                            [c + d / 2, c, c]),
                 cell=(a, a, a))

calc.set(txt='H2.out')
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
calc.write('H2.gpw')

print 'hydrogen atom energy:     %5.2f eV' % e1
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)
