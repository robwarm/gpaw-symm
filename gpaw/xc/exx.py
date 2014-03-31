# Todo: ACDF formula
from __future__ import division

import sys
from math import pi

import numpy as np
from ase.units import Hartree, Bohr
from ase.utils import prnt

import gpaw.mpi as mpi
from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.utilities import unpack, unpack2, packed_index, erf


def pawexxvv(atomdata, D_ii):
    """PAW correction for valence-valence EXX energy."""
    ni = len(D_ii)
    V_ii = np.empty((ni, ni))
    for i1 in range(ni):
        for i2 in range(ni):
            V = 0.0
            for i3 in range(ni):
                p13 = packed_index(i1, i3, ni)
                for i4 in range(ni):
                    p24 = packed_index(i2, i4, ni)
                    V += atomdata.M_pp[p13, p24] * D_ii[i3, i4]
            V_ii[i1, i2] = V
    return V_ii

        
class EXX(PairDensity):
    def __init__(self, calc, xc=None, kpts=None, bands=None, ecut=150.0,
                 alpha=0.0, skip_gamma=False,
                 world=mpi.world, txt=sys.stdout):
    
        alpha /= Bohr**2

        PairDensity.__init__(self, calc, ecut, world=world, txt=txt)

        ecut /= Hartree
        
        if xc is None:
            self.exx_fraction = 1.0
            xc = XC(XCNull())
        if xc == 'PBE0':
            self.exx_fraction = 0.25
            xc = XC('HYB_GGA_XC_PBEH')
        elif xc == 'B3LYP':
            self.exx_fraction = 0.2
            xc = XC('HYB_GGA_XC_B3LYP')
        self.xc = xc
        self.exc = np.nan  # density dependent part of xc-energy
        
        if kpts is None:
            # Do all k-points in the IBZ:
            kpts = range(self.calc.wfs.kd.nibzkpts)
        
        if bands is None:
            # Do all occupied bands:
            bands = [0, self.nocc2]
        
        prnt('Calculating exact exchange contributions for band index',
             '%d-%d' % (bands[0], bands[1] - 1), file=self.fd)
        prnt('for IBZ k-points with indices:',
             ', '.join(str(i) for i in kpts), file=self.fd)
        
        self.kpts = kpts
        self.bands = bands

        shape = (self.calc.wfs.nspins, len(kpts), bands[1] - bands[0])
        self.exxvv_sin = np.zeros(shape)   # valence-valence exchange energies
        self.exxvc_sin = np.zeros(shape)   # valence-core exchange energies
        self.f_sin = np.empty(shape)       # occupation numbers

        # The total EXX energy will not be calculated if we are only
        # interested in a few eigenvalues for a few k-points
        self.exx = np.nan    # total EXX energy
        self.exxvv = np.nan  # valence-valence
        self.exxvc = np.nan  # valence-core
        self.exxcc = 0.0     # core-core

        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(self.nocc2)
        
        # All occupied states:
        self.mykpts = [self.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]

        # Compensation charge used for molecular calculations only:
        self.beta = None      # e^(-beta*r^2)
        self.ngauss_G = None  # density
        self.vgauss_G = None  # potential

        self.G0 = None  # effective value for |G+q| when |G+q|=0
        
        self.skip_gamma = skip_gamma
        
        if not self.calc.atoms.pbc.any():
            # Set exponent of exp-function to -19 on the boundary:
            self.beta = 4 * 19 * (self.calc.wfs.gd.icell_cv**2).sum(1).max()
            prnt('Gaussian for electrostatic decoupling: e^(-beta*r^2),',
                 'beta=%.3f 1/Ang^2' % (self.beta / Bohr**2), file=self.fd)
        elif skip_gamma:
            prnt('Skip |G+q|=0 term', file=self.fd)
        else:
            # Volume per q-point:
            dvq = (2 * pi)**3 / self.vol / self.calc.wfs.kd.nbzkpts
            qcut = (dvq / (4 * pi / 3))**(1 / 3)
            if alpha == 0.0:
                self.G0 = (4 * pi * qcut / dvq)**-0.5
            else:
                self.G0 = (2 * pi**1.5 * erf(alpha**0.5 * qcut) / alpha**0.5 /
                           dvq)**-0.5
            prnt('G+q=0 term: Integrate e^(-alpha*q^2)/q^2 for',
                 'q<%.3f 1/Ang and alpha=%.3f Ang^2' %
                 (qcut / Bohr, alpha * Bohr**2), file=self.fd)

        # PAW matrices:
        self.V_asii = []  # valence-valence correction
        self.C_aii = []   # valence-core correction
        self.initialize_paw_exx_corrections()
        
    def calculate(self):
        kd = self.calc.wfs.kd
        nspins = self.calc.wfs.nspins
        
        for s in range(nspins):
            for i, k1 in enumerate(self.kpts):
                K1 = kd.ibz2bz_k[k1]
                kpt1 = self.get_k_point(s, K1, *self.bands)
                self.f_sin[s, i] = kpt1.f_n
                for kpt2 in self.mykpts:
                    if kpt2.s == s:
                        self.calculate_q(i, kpt1, kpt2)
                
                self.calculate_paw_exx_corrections(i, kpt1)

        self.world.sum(self.exxvv_sin)
        
        # Calculate total energy if we have everything needed:
        if (len(self.kpts) == kd.nibzkpts and
            self.bands[0] == 0 and
            self.bands[1] >= self.nocc2):
            exxvv_i = (self.exxvv_sin * self.f_sin).sum(axis=2).sum(axis=0)
            exxvc_i = 2 * (self.exxvc_sin * self.f_sin).sum(axis=2).sum(axis=0)
            self.exxvv = np.dot(kd.weight_k[self.kpts], exxvv_i) / nspins
            self.exxvc = np.dot(kd.weight_k[self.kpts], exxvc_i) / nspins
            self.exx = self.exxvv + self.exxvc + self.exxcc
            prnt('Exact exchange energy:', file=self.fd)
            for kind, exx in [('valence-valence', self.exxvv),
                              ('valence-core', self.exxvc),
                              ('core-core', self.exxcc),
                              ('total', self.exx)]:
                prnt('%16s%11.3f eV' % (kind + ':', exx * Hartree),
                     file=self.fd)
            
            self.exc = self.calculate_hybrid_correction()

        exx_sin = self.exxvv_sin + self.exxvc_sin
        prnt('EXX eigenvalue contributions in eV:', file=self.fd)
        prnt(np.array_str(exx_sin * Hartree, precision=3), file=self.fd)
    
    def get_exx_energy(self):
        return self.exx * Hartree
    
    def get_total_energy(self):
        ham = self.calc.hamiltonian
        return (self.exx * self.exx_fraction + self.exc +
                ham.Etot - ham.Exc) * Hartree
        
    def get_eigenvalue_contributions(self):
        return (self.exxvv_sin + self.exxvc_sin) * self.exx_fraction * Hartree
        
    def calculate_q(self, i, kpt1, kpt2):
        wfs = self.calc.wfs
        q_c = wfs.kd.bzk_kc[kpt2.K] - wfs.kd.bzk_kc[kpt1.K]
        if self.skip_gamma and not q_c.any():
            return
            
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, wfs.dtype, kd=qd)
        Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                   kpt1.shift_c - kpt2.shift_c)

        Q_aGii = self.initialize_paw_corrections(pd, soft=True)
        
        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                     for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                 pd, Q_G)
            e = self.calculate_n(pd, n, n_mG, kpt2)
            self.exxvv_sin[kpt1.s, i, n] += e

    def calculate_n(self, pd, n, n_mG, kpt2):
        molecule = not self.calc.atoms.pbc.any()
        
        G2_G = pd.G2_qG[0]
        iG_G = np.empty(len(G2_G))
        iG_G[1:] = G2_G[1:]**-0.5
        
        if G2_G[0] == 0.0:
            if molecule:
                iG_G[0] = 0.0
            else:
                iG_G[0] = 1 / self.G0
        else:
            iG_G[0] = G2_G[0]**-0.5

        e = 0.0
        f_m = kpt2.f_n
        
        if molecule and kpt2.n1 <= n < kpt2.n2:
            if self.ngauss_G is None:
                self.initialize_gaussian_compensation_charge(pd)
            m = n - kpt2.n1
            n_mG[m] -= self.ngauss_G
            e -= 2 * f_m[m] * (pd.integrate(self.vgauss_G, n_mG[m]) +
                               (self.beta / 2 / pi)**0.5)

        x = 4 * pi / self.calc.wfs.kd.nbzkpts / pd.gd.dv**2
        for f, n_G in zip(f_m, n_mG):
            x_G = n_G * iG_G
            e -= x * f * pd.integrate(x_G, x_G)

        return e

    def initialize_paw_exx_corrections(self):
        for a, atomdata in enumerate(self.calc.wfs.setups):
            V_sii = []
            for D_p in self.calc.density.D_asp[a]:
                D_ii = unpack2(D_p)
                V_ii = pawexxvv(atomdata, D_ii)
                V_sii.append(V_ii)
            C_ii = unpack(atomdata.X_p)
            self.V_asii.append(V_sii)
            self.C_aii.append(C_ii)
            self.exxcc += atomdata.ExxC

    def calculate_paw_exx_corrections(self, i, kpt):
        x = self.calc.wfs.nspins / self.world.size
        s = kpt.s
        
        for V_sii, C_ii, P_ni in zip(self.V_asii, self.C_aii, kpt.P_ani):
            V_ii = V_sii[s]
            v_n = (np.dot(P_ni, V_ii) * P_ni.conj()).sum(axis=1).real
            c_n = (np.dot(P_ni, C_ii) * P_ni.conj()).sum(axis=1).real
            self.exxvv_sin[s, i] -= v_n * x
            self.exxvc_sin[s, i] -= c_n

    def calculate_hybrid_correction(self):
        dens = self.calc.density
        if dens.nt_sg is None:
            dens.interpolate_pseudo_density()
        exc = self.xc.calculate(dens.finegd, dens.nt_sg)
        for a, D_sp in dens.D_asp.items():
            atomdata = dens.setups[a]
            exc += self.xc.calculate_paw_correction(atomdata, D_sp)
        return exc

    def initialize_gaussian_compensation_charge(self, pd):
        """Calculate gaussian compensation charge and its potential.

        Used to decouple electrostatic interactions between
        periodically repeated images for molecular calculations.

        Charge containing one electron::

            (beta/pi)^(3/2)*exp(-beta*r^2),

        its Fourier transform::

            exp(-G^2/(4*beta)),

        and its potential::

            erf(beta^0.5*r)/r.
        """

        gd = pd.gd

        # Calculate gaussian:
        G_Gv = pd.G_Qv[pd.Q_qG[0]]
        G2_G = pd.G2_qG[0]
        C_v = gd.cell_cv.sum(0) / 2  # center of cell
        self.ngauss_G = np.exp(-1.0 / (4 * self.beta) * G2_G +
                               1j * np.dot(G_Gv, C_v))

        # Calculate potential from gaussian:
        R_Rv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        r_R = ((R_Rv - C_v)**2).sum(3)**0.5
        if (gd.N_c % 2 == 0).all():
            r_R[tuple(gd.N_c // 2)] = 1.0  # avoid dividing by zero
        v_R = erf(self.beta**0.5 * r_R) / r_R
        if (gd.N_c % 2 == 0).all():
            v_R[tuple(gd.N_c // 2)] = (4 * self.beta / pi)**0.5
        self.vgauss_G = pd.fft(v_R) / gd.dv

        # Compare self-interaction to analytic result:
        assert abs(0.5 * pd.integrate(self.ngauss_G, self.vgauss_G) -
                   (self.beta / 2 / pi)**0.5) < 1e-6
