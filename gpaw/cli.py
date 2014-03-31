from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac, MethfesselPaxton
from gpaw.mixer import Mixer, MixerSum, MixerDif
from gpaw.poisson import PoissonSolver
from gpaw.eigensolvers import RMM_DIIS
from ase.cli.run import Runner, main as ase_main


def main():
    runner = Runner()
    runner.calculator_name = 'gpaw'
    runner.parameter_namespace = {
        'PW': PW,
        'FermiDirac': FermiDirac,
        'MethfesselPaxton': MethfesselPaxton,
        'Mixer': Mixer,
        'MixerSum': MixerSum,
        'MixerDif': MixerDif,
        'PoissonSolver': PoissonSolver,
        'RMM_DIIS': RMM_DIIS}
    ase_main(runner)
