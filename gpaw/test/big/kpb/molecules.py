"""Test exact exchange for 20 small molecules.

Compare results to::

  S. Kurth, J. P. Perdew, and P. Blaha
  Molecular and Soild-State Tests of Density Functional
  Approximations: LSD, GGAs, and Meta-GGAs
  International Journal of Quantum Chemistry, Vol. 75, 889-909, 1999

"""

from ase.data.g2_1_ref import diatomic, ex_atomization
from ase.tasks.molecule import MoleculeTask
from ase import Atoms

from gpaw import PW
from gpaw.factory import GPAWFactory
from gpaw.xc.hybridg import HybridXC


bondlengths = {'H2': 0.741,
               'OH': 0.970,
               'HF': 0.9168,
               'Be2': 2.460,
               'NO': 1.154,
               'P2': 1.893}
bondlengths.update((name, d[0]) for name, d in diatomic.items())

extra = {
   'CH4': ('CH4', [(0.0000, 0.0000, 0.0000),
                   (0.6276, 0.6276, 0.6276),
                   (0.6276, -0.6276, -0.6276),
                   (-0.6276, 0.6276, -0.6276),
                   (-0.6276, -0.6276, 0.6276)]),
   'NH3': ('NH3', [(0.0000, 0.0000, 0.0000),
                   (0.0000, -0.9377, -0.3816),
                   (0.8121,  0.4689, -0.3816),
                   (-0.8121, 0.4689, -0.3816)]),
   'H2O': ('OH2', [(0.0000, 0.0000, 0.1173),
                   (0.0000, 0.7572, -0.4692),
                   (0.0000, -0.7572, -0.4692)]),
   'C2H2': ('C2H2', [(0.0000, 0.0000, 0.6013),
                     (0.0000, 0.0000, -0.6013),
                     (0.0000, 0.0000, 1.6644),
                     (0.0000, 0.0000, -1.6644)]),
   'C2H4': ('C2H4', [(0.0000, 0.0000, 0.6695),
                     (0.0000, 0.0000, -0.6695),
                     (0.0000, 0.9289, 1.2321),
                     (0.0000, -0.9289, 1.2321),
                     (0.0000, 0.9289, -1.2321),
                     (0.0000, -0.9289, -1.2321)]),
   'HCN': ('CHN', [(0.0000, 0.0000, 0.0000),
                   (0.0000, 0.0000, 1.0640),
                   (0.0000, 0.0000, -1.1560)])}


class KurthPerdewBlahaMolecules(MoleculeTask):
    def run(self):
        return MoleculeTask.run(self, ex_atomization.keys())

    def build_system(self, name):
        if name in extra:
            mol = Atoms(*extra[name])
            mol.cell = self.unit_cell
            mol.center()
        else:
            self.bond_length = bondlengths.get(name, None)
            mol = MoleculeTask.build_system(self, name)
        return mol
   
    def calculate(self, name, atoms):
        data = MoleculeTask.calculate(self, name, atoms)
        exx = HybridXC('EXX', alpha=5.0)
        dexx = atoms.calc.get_xc_difference(exx)
        data['EXX'] = dexx
        data['EXXvc'] = exx.evc
        return data
    

task = KurthPerdewBlahaMolecules(
    calcfactory=GPAWFactory(xc='PBE', 
                            mode=PW(500)),
    tag='pw',
    cell=(11, 12, 13),
    atomize=True,
    use_lock_files=True)

task.run()
