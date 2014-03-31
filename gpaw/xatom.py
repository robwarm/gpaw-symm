import ase.units as units
import numpy as np

from gpaw.setup import BaseSetup
from gpaw.spline import Spline
from gpaw.basis_data import Basis


class X(BaseSetup):
    def __init__(self, e0=-10.0, rc=2.5):
        e0 /= units.Hartree
        rc /= units.Bohr
        self.e0 = e0
        self.rc = rc
        
        self.E = 0.0
        self.Z = 1
        self.Nc = 0
        self.Nv = 1
        self.nao = None
        self.pt_j = []
        self.ni = 0
        self.l_j = []
        self.nct = Spline(0, 0.5, [0.0, 0.0, 0.0])
        self.Nct = 0.0

        r2_g = np.linspace(0, rc, 100)**2
        alpha = 10.0 / rc**2
        x_g = np.exp(-alpha * r2_g)
        self.ghat_l = [Spline(0, rc, 4 * alpha**1.5 / np.pi**0.5 * x_g)]
        
        x_g = np.linspace(0, 1, 100)
        self.vbar = Spline(0, rc, e0 * (1 - 3 * x_g**2 + 2 * x_g**3))
        self.Delta_pL = np.zeros((0, 1))
        self.Delta0 = -1 / (4 * np.pi)**0.5
        self.lmax = 0
        self.K_p = self.M_p = self.MB_p = np.zeros(0)
        self.M_pp = np.zeros((0, 0))
        self.Kc = 0.0
        self.MB = 0.0
        self.M = 0.0
        self.xc_correction = None
        self.HubU = None
        self.dO_ii = np.zeros((0, 0))
        self.N0_p = np.zeros(0)
        self.type = 'x'
        self.fingerprint = None
        
        basis = Basis('H', 'sz(dzp)')
        self.basis = basis
        self.phit_j = self.basis.tosplines()
        self.f_j = []
        self.n_j = []
        self.nao = self.basis.nao
    
    def print_info(self, text):
        text('X: local potential: ' +
             'v=e0*(1-3x^2+2x^3), e0=%.3f eV, x=r/%.3f Ang' %
             (self.e0 * units.Hartree, self.rc * units.Bohr))
        
    def calculate_initial_occupation_numbers(self, magmom, hund, charge,
                                             nspins):
        if nspins == 1:
            return np.array([[1.0]])
        else:
            return np.array([[1.0], [0.0]])
