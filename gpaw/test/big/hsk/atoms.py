import sys

from ase.tasks.molecule import MoleculeTask

from gpaw import PW
from gpaw.factory import GPAWFactory


setups = sys.argv[1]

for ecut in [400, 500, 600]:
    task = MoleculeTask(
            calcfactory=GPAWFactory(xc='PBE',
                                    setups=setups,
                                    mode=PW(ecut)),
            tag='%s-%d' % (setups, ecut),
            cell=(11, 12, 13),
            use_lock_files=True)
    task.run(['C', 'Si', 'Ge', 'Al', 'N', 'P', 'As', 'Ga', 'In', 'Sb', 'Mg', 'O',
              'Li', 'F', 'Na', 'Cl', 'Cu', 'Rh', 'Pd', 'Ag'])
