import sys

from ase.dft.kpoints import monkhorst_pack

from gpaw import PW
from gpaw.factory import GPAWFactory
from gpaw.test.big.hsk.hsk import HarlSchimkaKressePBEBulkTask as PBEBulkTask


setups = sys.argv[1]

for ecut in [400, 500, 600]:
    for k in [10, 12, 14]:
        kpts = monkhorst_pack((k, k, k)) + 0.5 / k
        task = PBEBulkTask(
            calcfactory=GPAWFactory(xc='PBE', 
                                    mode=PW(ecut),
                                    setups=setups,
                                    kpts=kpts),
            tag='%s-%d-%d' % (setups, ecut, k),
            use_lock_files=True)
        task.run()
