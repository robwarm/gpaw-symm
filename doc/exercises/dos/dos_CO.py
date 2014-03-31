import pylab as p
from gpaw import GPAW

calc = GPAW('CO.gpw', txt=None)
ehomo = calc.get_homo_lumo()[0]
energies, dos = calc.get_dos(width=.1)
p.plot(energies - ehomo, dos)
p.axis('tight')
p.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
p.ylabel('Density of States (1/eV)')
p.show()
