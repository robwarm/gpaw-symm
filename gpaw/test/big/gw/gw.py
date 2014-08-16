import ase.db
from ase.lattice import bulk

from gpaw import GPAW, PW
from gpaw.response.g0w0 import G0W0


data = {
    'C': ['diamond', 3.553],
    'Si': ['diamond', 5.421],
    'Ge': ['diamond', 5.644],
    'SiC': ['zincblende', 4.346],
    'AlN': ['zincblende', 4.368],
    'AlP': ['zincblende', 5.451],
    'AlAs': ['zincblende', 5.649],
    'GaN': ['zincblende', 4.520],
    'GaP': ['zincblende', 5.439],
    'GaAs': ['zincblende', 5.640],
    'InP': ['zincblende', 5.858],
    'InAs': ['zincblende', 6.047],
    'InSb': ['zincblende', 6.468]}


c = ase.db.connect('gw.db')

for name in data:
    id = c.reserve(name=name)
    if id is None:
        continue
        
    x, a = data[name]
    atoms = bulk(name, x, a=a)
    atoms.calc = GPAW(mode=PW(400),
                      kpts={'size': (6, 6, 6), 'gamma': True},
                      txt='%s.txt' % name)
    atoms.get_potential_energy()
    atoms.calc.diagonalize_full_hamiltonian(nbands=100)
    atoms.calc.write(name, mode='all')
    n = int(atoms.calc.get_number_of_electrons()) // 2
    gw = G0W0(name, 'gw-' + name,
              nbands=100,
              kpts=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
              ecut=150,
              hilbert=True,
              fast=True,
              domega0=0.1,
              eta=0.2,
              bands=(n - 1, n + 1))
    results = gw.calculate()
    c.write(atoms, name=name, data=results)
    del c[id]
