from ase.parallel import rank
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.xc_functional import XCFunctional

"""Test generation of setups

Run this script to generate exactly those setups needed for running
the tests.
"""

if rank == 0:
    files = []

    def gen(symbol, xcname):
        g = Generator(symbol, xcname, scalarrel=True, nofiles=True)
        g.run(exx=True, **parameters[symbol])
        files.append('%s.%s' % (symbol, XCFunctional(xcname).get_name()))

    for symbol in ['H', 'He', 'C', 'N', 'O', 'Cl', 'Al', 'Si', 'Na', 'Fe',
                   'Cu']:
        gen(symbol, 'X-C_PW')
    for symbol in ['H', 'He', 'Li', 'N']:
        gen(symbol, 'X_PBE-C_PBE')
    for symbol in ['He', 'Li']:
        gen(symbol, 'X_PBE_R-C_PBE')
    for symbol in ['Li']:
        gen(symbol, 'oldLDA')
    ##for symbol in ['Mg']:
    ##    gen(symbol, 'GLLB')

    if __name__ != '__main__' and rank == 0:
        # We have been imported by test.py, so we should clean up:
        from os import remove
        for file in files:
            remove(file)
