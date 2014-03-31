#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from math import pi, exp, sqrt, log

import numpy as np
from scipy.optimize import fsolve
from scipy import __version__ as scipy_version
from ase.utils import prnt, devnull
from ase.units import Hartree
from ase.data import atomic_numbers, chemical_symbols

from gpaw.spline import Spline
from gpaw.setup import BaseSetup
from gpaw.version import version
from gpaw.basis_data import Basis
from gpaw.gaunt import make_gaunt
from gpaw.utilities import erf, pack2
from gpaw.setup_data import SetupData
from gpaw.atom.configurations import configurations
from gpaw.utilities.lapack import general_diagonalize
from gpaw.atom.aeatom import AllElectronAtom, Channel, parse_ld_str, colors, \
    GaussianBasis


parameters = {
'H':  ('1s,s,p', 0.9, {}),
'He': ('1s,s,p', 1.5, {}),
'Li': ('1s,2s,2p,p,d', 1.5, {}),
'Be': ('1s,2s,2p,p,d', 1.4, {}),
'B':  ('2s,s,2p,p,d', 1.2, {}),
'C':  ('2s,s,2p,p,d', 1.2, {}),
'N':  ('2s,s,2p,p,d', [1.2, 1.3], {'r0': 1.1}),
'O':  ('2s,s,2p,p,d', [1.2, 1.4], {}),
'F':  ('2s,s,2p,p,d', [1.2,1.4], {}),
'Ne': ('2s,s,2p,p,d', 1.8, {}),  # 10
'Na': ('2s,3s,2p,3p,d', 2.3, {'local': 'f'}),
'Mg': ('2s,3s,2p,3p,d', [2.0, 1.8], {'local': 'f'}),
'Al': ('3s,s,3p,p,d', 2.1, {'local': 'f'}),
'Si': ('3s,s,3p,p,d', 1.9, {'local': 'f'}),
'P':  ('3s,s,3p,p,d', 1.7, {'local': 'f'}),
'S':  ('3s,s,3p,p,d', 1.6, {'local': 'f'}),
'Cl': ('3s,s,3p,p,d', 1.5, {'local': 'f'}),
'Ar': ('3s,s,3p,p,d', 1.5, {'local': 'f'}),
'K':  ('3s,4s,3p,4p,d,d', 2.1, {'local': 'f'}),
'Ca': ('3s,4s,3p,4p,3d,d', 2.1, {'local': 'f'}),  # 20
'Sc': ('3s,4s,3p,4p,3d,d', 2.3, {'local': 'f'}),
'Ti': ('3s,4s,3p,4p,3d,d', [2.2, 2.2, 2.3], {'local': 'f'}),
'V':  ('3s,4s,3p,4p,3d,d', [2.1, 2.1, 2.3], {'local': 'f'}),
'Cr': ('3s,4s,3p,4p,3d,d', [2.1, 2.1, 2.3], {'local': 'f'}),
'Mn': ('3s,4s,3p,4p,3d,d', [2.0, 2.0, 2.2], {'local': 'f'}),
'Fe': ('3s,4s,3p,4p,3d,d', 2.1, {'local': 'f'}),
'Co': ('3s,4s,3p,4p,3d,d', 2.1, {'local': 'f'}),
'Ni': ('3s,4s,3p,4p,3d,d', 2.0, {'local': 'f'}),
'Cu': ('3s,4s,3p,4p,3d,d', 1.9, {'local': 'f'}),
'Zn': ('3s,4s,3p,4p,3d,d', 1.9, {'local': 'f'}),  # 30
'Ga': ('4s,s,4p,p,3d,d', 2.2, {'local': 'f'}),
'Ge': ('4s,s,4p,p,3d,d', 2.1, {'local': 'f'}),
'As': ('4s,s,4p,p,d', 2.0, {'local': 'f'}),
'Se': ('4s,s,4p,p,d', 2.1, {'local': 'f'}),
'Br': ('4s,s,4p,p,d', 2.1, {'local': 'f'}),
'Kr': ('4s,s,4p,p,d', 2.1, {'local': 'f'}),
'Rb': ('4s,5s,4p,5p,d,d', 2.5, {'local': 'f'}),
'Sr': ('4s,5s,4p,5p,4d,d', 2.5, {'local': 'f'}),
'Y':  ('4s,5s,4p,5p,4d,d', 2.5, {'local': 'f'}),
'Zr': ('4s,5s,4p,5p,4d,d', 2.5, {'local': 'f'}),  # 40
'Nb': ('4s,5s,4p,5p,4d,d', [2.4,2.4,2.5], {'local': 'f'}),
'Mo': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Tc': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Ru': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Rh': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Pd': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Ag': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'Cd': ('4s,5s,4p,5p,4d,d', 2.3, {'local': 'f'}),
'In': ('5s,s,5p,p,4d,d', 2.6, {'local': 'f'}),
'Sn': ('5s,s,5p,p,4d,d', 2.5, {'local': 'f'}),
'Sb': ('5s,s,5p,p,4d,d', 2.5, {'local': 'f'}),
'Te': ('5s,6s,5p,p,d,d', 2.5, {'local': 'f'}),
'I':  ('5s,s,5p,p,d', 2.4, {'local': 'f'}),
'Xe': ('5s,s,5p,p,d', 2.3, {'local': 'f'}),
'Cs': ('5s,6s,5p,6p,5d', [1.9, 2.2], {}),  # 55
'Ba': ('5s,6s,5p,6p,5d', [1.8, 2.2], {}),
'La': ('5s,6s,5p,6p,5d,d,4f,f', 2.5, {'local': 'g'}),
'Ce': ('5s,6s,5p,6p,5d,d,4f,f', 2.4, {'local': 'g'}),
'Pr': ('5s,6s,5p,6p,5d,d,4f,f', 2.3, {'local': 'g'}),
'Nd': ('5s,6s,5p,6p,5d,d,4f,f', 2.3, {'local': 'g'}),  # 60
'Pm': ('5s,6s,5p,6p,5d,d,4f,f', 2.3, {'local': 'g'}),
'Sm': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Eu': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Gd': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Tb': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),  # 65
'Dy': ('5s,6s,5p,6p,5d,d,4f,f', 2.1, {'local': 'g'}),
'Ho': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Er': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Tm': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Yb': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),  # 70
'Lu': ('5s,6s,5p,6p,5d,d,4f,f', 2.2, {'local': 'g'}),
'Hf': ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),
'Ta': ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),
'W':  ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),
'Re': ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),  # 75
'Os': ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),
'Ir': ('5s,6s,5p,6p,5d,d', 2.4, {'local': 'f'}),
'Pt': ('5s,6s,5p,6p,5d,d', 2.3, {'local': 'f'}),
'Au': ('5s,6s,5p,6p,5d,d', 2.3, {'local': 'f'}),
'Hg': ('5s,6s,5p,6p,5d,d', 2.3, {'local': 'f'}),  # 80
'Tl': ('6s,s,6p,p,5d,d', 2.8, {'local': 'f'}),
'Pb': ('6s,s,6p,p,5d,d', 2.6, {'local': 'f'}),
'Bi': ('6s,s,6p,p,5d,d', 2.6, {'local': 'f'}),
'Po': ('6s,s,6p,p,d', 2.7, {'local': 'f'}),
'At': ('6s,s,6p,p,d', 2.6, {'local': 'f'}),
'Rn': ('6s,s,6p,p,d', 2.6, {'local': 'f'}),
}


class PAWWaves:
    def __init__(self, rgd, l, rcut):
        self.rgd = rgd
        self.l = l
        self.rcut = rcut

        self.n_n = []
        self.e_n = []
        self.f_n = []
        self.phi_ng = []
        self.phit_ng = None
        self.pt_ng = None

    def __len__(self):
        return len(self.n_n)

    def add(self, phi_g, n, e, f):
        self.phi_ng.append(phi_g)
        self.n_n.append(n)
        self.e_n.append(e)
        self.f_n.append(f)

    def pseudize(self, type, nderiv):
        rgd = self.rgd

        if type == 'poly':
            ps = rgd.pseudize
        elif type == 'bessel':
            ps = rgd.jpseudize

        phi_ng = self.phi_ng = np.array(self.phi_ng)
        N = len(phi_ng)
        phit_ng = self.phit_ng = rgd.empty(N)
        gcut = rgd.ceil(self.rcut)

        self.nt_g = 0
        self.c_n = []
        for n in range(N):
            phit_ng[n], c = ps(phi_ng[n], gcut, self.l, nderiv)
            self.c_n.append(c)
            self.nt_g += self.f_n[n] / 4 / pi * phit_ng[n]**2

        self.dS_nn = np.empty((N, N))
        for n1 in range(N):
            for n2 in range(N):
                self.dS_nn[n1, n2] = rgd.integrate(
                    phi_ng[n1] * phi_ng[n2] -
                    phit_ng[n1] * phit_ng[n2]) / (4 * pi)
        self.Q = np.dot(self.f_n, self.dS_nn.diagonal())

    def construct_projectors(self, vtr_g, rcmax):
        N = len(self)
        if N == 0:
            self.pt_ng = []
            return

        rgd = self.rgd
        phit_ng = self.phit_ng
        gcmax = rgd.ceil(rcmax)
        r_g = rgd.r_g
        l = self.l

        dgdr_g = 1 / rgd.dr_g
        d2gdr2_g = rgd.d2gdr2()

        q_ng = rgd.zeros(N)
        for n in range(N):
            a_g, dadg_g, d2adg2_g = rgd.zeros(3)
            a_g[1:] = self.phit_ng[n, 1:] / r_g[1:]**l
            a_g[0] = self.c_n[n]
            dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
            d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
            q_g = (vtr_g - self.e_n[n] * r_g) * self.phit_ng[n]
            q_g -= 0.5 * r_g**l * (
                (2 * (l + 1) * dgdr_g + r_g * d2gdr2_g) * dadg_g +
                r_g * d2adg2_g * dgdr_g**2)
            q_g[gcmax:] = 0
            q_g[1:] /= r_g[1:]
            if l == 0:
                q_g[0] = q_g[1]
            q_ng[n] = q_g

        A_nn = rgd.integrate(phit_ng[:, None] * q_ng) / (4 * pi)
        self.dH_nn = self.e_n * self.dS_nn - A_nn

        L_nn = np.eye(N)
        U_nn = A_nn.copy()

        if N - self.n_n.count(-1) == 1:
            assert self.n_n[0] != -1
            # We have a single bound-state projector.
            for n1 in range(N):
                for n2 in range(n1 + 1, N):
                    L_nn[n2, n1] = U_nn[n2, n1] / U_nn[n1, n1]
                    U_nn[n2] -= U_nn[n1] * L_nn[n2, n1]

            iL_nn = np.linalg.inv(L_nn)
            phit_ng[:] = np.dot(iL_nn, phit_ng)
            self.phi_ng[:] = np.dot(iL_nn, self.phi_ng)

            self.dS_nn = np.dot(np.dot(iL_nn, self.dS_nn), iL_nn.T)
            self.dH_nn = np.dot(np.dot(iL_nn, self.dH_nn), iL_nn.T)

        self.pt_ng = np.dot(np.linalg.inv(U_nn.T), q_ng)

    def calculate_kinetic_energy_correction(self, vr_g, vtr_g):
        if len(self) == 0:
            return
        self.dekin_nn = (self.rgd.integrate(self.phit_ng[:, None] *
                                            self.phit_ng *
                                            vtr_g, -1) / (4 * pi) -
                         self.rgd.integrate(self.phi_ng[:, None] *
                                            self.phi_ng *
                                            vr_g, -1) / (4 * pi) +
                         self.dH_nn)


class PAWSetupGenerator:
    def __init__(self, aea, projectors,
                 scalar_relativistic=False,
                 fd=sys.stdout):
        """fd: stream
            Text output."""

        self.aea = aea

        if fd is None:
            fd = devnull
        self.fd = fd

        self.lmax = -1
        self.states = {}
        for s in projectors.split(','):
            l = 'spdf'.find(s[-1])
            if len(s) == 1:
                n = None
            elif '.' in s:
                n = float(s[:-1])
            else:
                n = int(s[:-1])
            if l in self.states:
                self.states[l].append(n)
            else:
                self.states[l] = [n]
            if l > self.lmax:
                self.lmax = l

        # Add empty bound states:
        for l, nn in self.states.items():
            for n in nn:
                if (isinstance(n, int) and
                    (l not in aea.f_lsn or n - l > len(aea.f_lsn[l][0]))):
                    aea.add(n, l, 0)

        for l in range(self.lmax):
            if l not in self.states:
                states[l] = []

        aea.initialize()
        aea.run()
        aea.scalar_relativistic = scalar_relativistic
        aea.refine()

        self.rgd = aea.rgd

        self.vtr_g = None

        self.log('\nGenerating PAW', aea.xc.name, 'setup for', aea.symbol)

    def construct_shape_function(self, alpha=None, rc=None, eps=1e-10):
        """Build shape-function for compensation charge."""

        self.alpha = alpha

        if self.alpha is None:
            if isinstance(rc, list):
                rc = min(rc)
            rc = 1.5 * rc

            def spillage(alpha):
                """Fraction of gaussian charge outside rc."""
                x = alpha * rc**2
                return 1 - erf(sqrt(x)) + 2 * sqrt(x / pi) * exp(-x)

            def f(alpha):
                return log(spillage(alpha)) - log(eps)

            if scipy_version < '0.8':
                self.alpha = fsolve(f, 7.0)
            else:
                self.alpha = fsolve(f, 7.0)[0]

            self.alpha = round(self.alpha, 1)

        self.log('Shape function: exp(-alpha*r^2), alpha=%.1f Bohr^-2' %
                 self.alpha)

        self.ghat_g = (np.exp(-self.alpha * self.rgd.r_g**2) *
                       (self.alpha / pi)**1.5)

    def log(self, *args, **kwargs):
        prnt(file=self.fd, *args, **kwargs)

    def calculate_core_density(self):
        self.nc_g = self.rgd.zeros()
        self.ncore = 0
        self.nvalence = 0
        self.ekincore = 0.0
        for l, ch in enumerate(self.aea.channels):
            for n, f in enumerate(ch.f_n):
                if l <= self.lmax and n + l + 1 in self.states[l]:
                    self.nvalence += f
                else:
                    self.nc_g += f * ch.calculate_density(n)
                    self.ncore += f
                    self.ekincore += f * ch.e_n[n]

        self.ekincore -= self.rgd.integrate(self.nc_g * self.aea.vr_sg[0], -1)

        self.log('Core electrons:', self.ncore)
        self.log('Valence electrons:', self.nvalence)

    def add_waves(self, rc):
        if isinstance(rc, float):
            radii = [rc]
        else:
            radii = rc

        self.rcmax = max(radii)

        if self.lmax >= 0:
            radii += [radii[-1]] * (self.lmax + 1 - len(radii))

        self.waves_l = []
        for l in range(self.lmax + 1):
            rcut = radii[l]
            waves = PAWWaves(self.rgd, l, rcut)
            e = -1.0
            for n in self.states[l]:
                if isinstance(n, int):
                    # Bound state:
                    ch = self.aea.channels[l]
                    e = ch.e_n[n - l - 1]
                    f = ch.f_n[n - l - 1]
                    phi_g = ch.phi_ng[n - l - 1]
                else:
                    if n is None:
                        e += 1.0
                    else:
                        e = n
                    n = -1
                    f = 0.0
                    phi_g = self.rgd.zeros()
                    gc = self.rgd.round(1.5 * rcut)
                    ch = Channel(l)
                    ch.integrate_outwards(phi_g, self.rgd, self.aea.vr_sg[0], gc, e,
                                          self.aea.scalar_relativistic)
                    phi_g[1:gc + 1] /= self.rgd.r_g[1:gc + 1]
                    if l == 0:
                        phi_g[0] = phi_g[1]
                    phi_g /= (self.rgd.integrate(phi_g**2) / (4*pi))**0.5

                waves.add(phi_g, n, e, f)
            self.waves_l.append(waves)

    def pseudize(self, type='poly', nderiv=6, rcore=None):
        self.Q = -self.aea.Z + self.ncore

        self.nt_g = self.rgd.zeros()
        for waves in self.waves_l:
            waves.pseudize(type, nderiv)
            self.nt_g += waves.nt_g
            self.Q += waves.Q

        if rcore is None:
            rcore = self.rcmax * 0.8
        else:
            assert rcore <= self.rcmax

        # Make sure pseudo density is monotonically decreasing:
        while 1:
            gcore = self.rgd.round(rcore)
            self.nct_g = self.rgd.pseudize(self.nc_g, gcore)[0]
            nt_g = self.nt_g + self.nct_g
            dntdr_g = self.rgd.derivative(nt_g)[:gcore]
            if dntdr_g.max() < 0.0:
                break
            rcore -= 0.01

        if 1:
            rcore *= 1.2
            print rcore, '1.200000000000000000000000000'
            gcore = self.rgd.round(rcore)
            self.nct_g = self.rgd.pseudize(self.nc_g, gcore)[0]
            nt_g = self.nt_g + self.nct_g

        self.log('Constructing smooth pseudo core density for r < %.3f' %
                 rcore)
        self.nt_g = nt_g

        if 0:
            # Constuct function that decrease smoothly from
            # f(0)=1 to f(rcmax)=0:
            x_g = self.rgd.r_g[:gcore] / self.rcmax
            f_g = self.rgd.zeros()
            f_g[:gcore] = (1 - x_g**2 * (3 - 2 * x_g))**2

            # Add enough of f to nct to make nt monotonically decreasing:
            dfdr_g = self.rgd.derivative(f_g)
            A = (-dntdr_g / dfdr_g[:gcore]).max() * 1.5
            self.nt_g += A * f_g
            self.nct_g += A * f_g
            self.log('Adding to nct ...')

        self.npseudocore = self.rgd.integrate(self.nct_g)
        self.log('Pseudo core electrons: %.6f' % self.npseudocore)
        self.Q -= self.npseudocore

        self.rhot_g = self.nt_g + self.Q * self.ghat_g
        self.vHtr_g = self.rgd.poisson(self.rhot_g)

        self.vxct_g = self.rgd.zeros()
        exct_g = self.rgd.zeros()
        self.exct = self.aea.xc.calculate_spherical(
            self.rgd, self.nt_g.reshape((1, -1)), self.vxct_g.reshape((1, -1)))

        self.v0r_g = self.vtr_g - self.vHtr_g - self.vxct_g * self.rgd.r_g
        self.v0r_g[self.rgd.round(self.rcmax):] = 0.0

        self.log('\nProjectors:')
        self.log(' state  occ         energy             norm        rcut')
        self.log(' nl            [Hartree]  [eV]      [electrons]   [Bohr]')
        self.log('----------------------------------------------------------')
        for l, waves in enumerate(self.waves_l):
            for n, e, f, ds in zip(waves.n_n, waves.e_n, waves.f_n,
                                  waves.dS_nn.diagonal()):
                if n == -1:
                    self.log('  %s         %10.6f %10.5f   %19.2f' %
                             ('spdf'[l], e, e * Hartree, waves.rcut))
                else:
                    self.log(
                        ' %d%s     %2d  %10.6f %10.5f      %5.3f  %9.2f' %
                             (n, 'spdf'[l], f, e, e * Hartree, 1 - ds,
                              waves.rcut))
        self.log()

    def find_local_potential(self, l0, r0, P, e0):
        if l0 is None:
            self.find_polynomial_potential(r0, P)
        else:
            self.match_local_potential(l0, r0, P, e0)

    def find_polynomial_potential(self, r0, P, e0=None):
        self.log('Constructing smooth local potential for r < %.3f' % r0)
        g0 = self.rgd.ceil(r0)
        assert e0 is None

        self.vtr_g = self.rgd.pseudize(self.aea.vr_sg[0], g0, 1, P)[0]

        self.l0 = None
        self.e0 = None
        self.r0 = r0
        self.nderiv0 = P

    def match_local_potential(self, l0, r0, P, e0):
        self.log('Local potential matching %s-scattering at e=%.3f eV' %
                 ('spdfg'[l0], e0 * Hartree) +
                 ' and r=%.2f Bohr' % r0)

        g0 = self.rgd.ceil(r0)
        gc = g0 + 20

        ch = Channel(l0)
        phi_g = self.rgd.zeros()
        ch.integrate_outwards(phi_g, self.rgd, self.aea.vr_sg[0], gc, e0,
                              self.aea.scalar_relativistic)
        phi_g[1:gc] /= self.rgd.r_g[1:gc]
        if l0 == 0:
            phi_g[0] = phi_g[1]

        #phit_g, c = self.rgd.pseudize_normalized(phi_g, g0, l=l0, points=P)
        phit_g, c = self.rgd.pseudize(phi_g, g0, l=l0, points=P)
        r_g = self.rgd.r_g[1:g0]

        dgdr_g = 1 / self.rgd.dr_g
        d2gdr2_g = self.rgd.d2gdr2()
        a_g = phit_g.copy()
        a_g[1:] /= self.rgd.r_g[1:]**l0
        a_g[0] = c
        dadg_g = self.rgd.zeros()
        d2adg2_g = self.rgd.zeros()
        dadg_g[1:-1] = 0.5 * (a_g[2:] - a_g[:-2])
        d2adg2_g[1:-1] = a_g[2:] - 2 * a_g[1:-1] + a_g[:-2]
        q_g = (((l0 + 1) * dgdr_g + 0.5 * self.rgd.r_g * d2gdr2_g) * dadg_g +
               0.5 * self.rgd.r_g * d2adg2_g * dgdr_g**2)
        q_g[:g0] /= a_g[:g0]
        q_g += e0 * self.rgd.r_g
        q_g[0] = 0.0

        self.vtr_g = self.aea.vr_sg[0].copy()
        self.vtr_g[0] = 0.0
        self.vtr_g[1:g0] = q_g[1:g0]#e0 * r_g - t_g * r_g**(l0 + 1) / phit_g[1:g0]
        self.l0 = l0
        self.e0 = e0
        self.r0 = r0
        self.nderiv0 = P

    def construct_projectors(self):
        for waves in self.waves_l:
            waves.construct_projectors(self.vtr_g, self.rcmax)
            waves.calculate_kinetic_energy_correction(self.aea.vr_sg[0],
                                                      self.vtr_g)

    def check_all(self):
        self.log(('Checking eigenvalues of %s pseudo atom using ' +
                  'a Gaussian basis set:') % self.aea.symbol)
        self.log('                 AE [eV]        PS [eV]      error [eV]')

        ok = True

        for l in range(4):
            try:
                e_b, n0 = self.check(l)
            except RuntimeError:
                self.log('Singular overlap matrix!')
                ok = False
                continue

            nbound = (e_b < -0.002).sum()

            if l < len(self.aea.channels):
                e0_b = self.aea.channels[l].e_n
                nbound0 = (e0_b < -0.002).sum()
                extra = 6
                for n in range(1 + l, nbound0 + 1 + l + extra):
                    if n - 1 - l < len(self.aea.channels[l].f_n):
                        f = self.aea.channels[l].f_n[n - 1 - l]
                        self.log('%2d%s  %2d' % (n, 'spdf'[l], f), end='')
                    else:
                        self.log('       ', end='')
                    self.log('  %15.3f' % (e0_b[n - 1 - l] * Hartree), end='')
                    if n - 1 - l - n0 >= 0:
                        self.log('%15.3f' * 2 %
                                 (e_b[n - 1 - l - n0] * Hartree,
                                  (e_b[n - 1 - l - n0] - e0_b[n - 1 - l]) *
                                  Hartree))
                    else:
                        self.log()

                if nbound != nbound0 - n0:
                    self.log('Wrong number of %s-states!' % 'spdf'[l])
                    ok = False
                elif (nbound > 0 and
                      abs(e_b[:nbound] - e0_b[n0:nbound0]).max() > 1e-3):
                    self.log('Error in bound %s-states!' % 'spdf'[l])
                    ok = False
                elif (abs(e_b[nbound:nbound + extra] -
                          e0_b[nbound0:nbound0 + extra]).max() > 2e-2):
                    self.log('Error in %s-states!' % 'spdf'[l])
                    ok = False
            elif nbound > 0:
                self.log('Wrong number of %s-states!' % 'spdf'[l])
                ok = False

        return ok

    def check(self, l):
        basis = self.aea.channels[0].basis
        eps = basis.eps
        alpha_B = basis.alpha_B

        basis = GaussianBasis(l, alpha_B, self.rgd, eps)
        H_bb = basis.calculate_potential_matrix(self.vtr_g)
        H_bb += basis.T_bb
        S_bb = np.eye(len(basis))

        n0 = 0
        if l < len(self.waves_l):
            waves = self.waves_l[l]
            if len(waves) > 0:
                P_bn = self.rgd.integrate(basis.basis_bg[:, None] *
                                          waves.pt_ng) / (4 * pi)
                H_bb += np.dot(np.dot(P_bn, waves.dH_nn), P_bn.T)
                S_bb += np.dot(np.dot(P_bn, waves.dS_nn), P_bn.T)
                n0 = waves.n_n[0] - l - 1
                if n0 < 0 and l < len(self.aea.channels):
                    n0 = (self.aea.channels[l].f_n > 0).sum()
        elif l < len(self.aea.channels):
            n0 = (self.aea.channels[l].f_n > 0).sum()

        e_b = np.empty(len(basis))
        general_diagonalize(H_bb, e_b, S_bb)
        return e_b, n0

    def test_convergence(self):
        rgd = self.rgd
        r_g = rgd.r_g
        G_k, nt_k = self.rgd.fft(self.nt_g * r_g)
        rhot_k = self.rgd.fft(self.rhot_g * r_g)[1]
        ghat_k = self.rgd.fft(self.ghat_g * r_g)[1]
        v0_k = self.rgd.fft(self.v0r_g)[1]
        vt_k = self.rgd.fft(self.vtr_g)[1]
        phi_k = self.rgd.fft(self.waves_l[0].phit_ng[0] * r_g)[1]
        eee_k = 0.5 * nt_k**2 * (4 * pi)**2 / (2 * pi)**3
        ecc_k = 0.5 * rhot_k**2 * (4 * pi)**2 / (2 * pi)**3
        egg_k = 0.5 * ghat_k**2 * (4 * pi)**2 / (2 * pi)**3
        ekin_k = 0.5 * phi_k**2 * G_k**4 / (2 * pi)**3
        evt_k = nt_k * vt_k * G_k**2 * 4 * pi / (2 * pi)**3

        eee = 0.5 * rgd.integrate(self.nt_g * rgd.poisson(self.nt_g), -1)
        ecc = 0.5 * rgd.integrate(self.rhot_g * self.vHtr_g, -1)
        egg = 0.5 * rgd.integrate(self.ghat_g * rgd.poisson(self.ghat_g), -1)
        ekin = self.aea.ekin - self.ekincore - self.waves_l[0].dekin_nn[0, 0]
        print self.aea.ekin, self.ekincore, self.waves_l[0].dekin_nn[0, 0]
        evt = rgd.integrate(self.nt_g * self.vtr_g, -1)

        import pylab as p

        errors = 10.0**np.arange(-4, 0) / Hartree
        self.log('\nConvergence of energy:')
        self.log('plane-wave cutoff (wave-length) [ev (Bohr)]\n  ', end='')
        for de in errors:
            self.log('%14.4f' % (de * Hartree), end='')
        for label, e_k, e0 in [
            ('e-e', eee_k, eee),
            ('c-c', ecc_k, ecc),
            ('g-g', egg_k, egg),
            ('kin', ekin_k, ekin),
            ('vt', evt_k, evt)]:
            self.log('\n%3s: ' % label, end='')
            e_k = (np.add.accumulate(e_k) - 0.5 * e_k[0] - 0.5 * e_k) * G_k[1]
            print e_k[-1],e0, e_k[-1]-e0
            k = len(e_k) - 1
            for de in errors:
                while abs(e_k[k] - e_k[-1]) < de:
                    k -= 1
                G = k * G_k[1]
                ecut = 0.5 * G**2
                h = pi / G
                self.log(' %6.1f (%4.2f)' % (ecut * Hartree, h), end='')
            p.semilogy(G_k, abs(e_k - e_k[-1]) * Hartree, label=label)
        self.log()
        p.axis(xmax=20)
        p.xlabel('G')
        p.ylabel('[eV]')
        p.legend()
        p.show()

    def plot(self):
        import matplotlib.pyplot as plt
        r_g = self.rgd.r_g

        plt.figure()
        plt.plot(r_g, self.vxct_g, label='xc')
        plt.plot(r_g[1:], self.v0r_g[1:] / r_g[1:], label='0')
        plt.plot(r_g[1:], self.vHtr_g[1:] / r_g[1:], label='H')
        plt.plot(r_g[1:], self.vtr_g[1:] / r_g[1:], label='ps')
        plt.plot(r_g[1:], self.aea.vr_sg[0, 1:] / r_g[1:], label='ae')
        plt.axis(xmax=2 * self.rcmax,
                 ymin=self.vtr_g[1] / r_g[1],
                 ymax=max(0, (self.v0r_g[1:] / r_g[1:]).max()))
        plt.xlabel('radius [Bohr]')
        plt.ylabel('potential [Ha]')
        plt.legend()

        plt.figure()
        i = 0
        for l, waves in enumerate(self.waves_l):
            for n, e, phi_g, phit_g in zip(waves.n_n, waves.e_n,
                                           waves.phi_ng, waves.phit_ng):
                if n == -1:
                    gc = self.rgd.ceil(waves.rcut)
                    name = '*%s (%.2f Ha)' % ('spdf'[l], e)
                else:
                    gc = len(self.rgd)
                    name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
                plt.plot(r_g[:gc], (phi_g * r_g)[:gc], color=colors[i],
                         label=name)
                plt.plot(r_g[:gc], (phit_g * r_g)[:gc], '--', color=colors[i])
                i += 1
        plt.axis(xmax=3 * self.rcmax)
        plt.xlabel('radius [Bohr]')
        plt.ylabel(r'$r\phi_{n\ell}(r)$')
        plt.legend()

        plt.figure()
        i = 0
        for l, waves in enumerate(self.waves_l):
            for n, e, pt_g in zip(waves.n_n, waves.e_n, waves.pt_ng):
                if n == -1:
                    name = '*%s (%.2f Ha)' % ('spdf'[l], e)
                else:
                    name = '%d%s (%.2f Ha)' % (n, 'spdf'[l], e)
                plt.plot(r_g, pt_g * r_g, color=colors[i], label=name)
                i += 1
        plt.axis(xmax=self.rcmax)
        plt.legend()

    def logarithmic_derivative(self, l, energies, rcut):
        rgd = self.rgd
        ch = Channel(l)
        gcut = rgd.round(rcut)

        N = 0
        if l < len(self.waves_l):
            # Nonlocal PAW stuff:
            waves = self.waves_l[l]
            if len(waves) > 0:
                pt_ng = waves.pt_ng
                dH_nn = waves.dH_nn
                dS_nn = waves.dS_nn
                N = len(pt_ng)

        u_g = rgd.zeros()
        u_ng = rgd.zeros(N)
        dudr_n = np.empty(N)

        logderivs = []
        for e in energies:
            dudr = ch.integrate_outwards(u_g, rgd, self.vtr_g, gcut, e)
            u = u_g[gcut]

            if N:
                for n in range(N):
                    dudr_n[n] = ch.integrate_outwards(u_ng[n], rgd,
                                                      self.vtr_g, gcut, e,
                                                      pt_g=pt_ng[n])

                A_nn = (dH_nn - e * dS_nn) / (4 * pi)
                B_nn = rgd.integrate(pt_ng[:, None] * u_ng, -1)
                c_n  = rgd.integrate(pt_ng * u_g, -1)
                d_n = np.linalg.solve(np.dot(A_nn, B_nn) + np.eye(N),
                                      np.dot(A_nn, c_n))
                u -= np.dot(u_ng[:, gcut], d_n)
                dudr -= np.dot(dudr_n, d_n)

            logderivs.append(dudr / u)

        return logderivs

    def make_paw_setup(self, tag=None):
        aea = self.aea

        setup = SetupData(aea.symbol, aea.xc.name, tag, readxml=False)

        nj = sum(len(waves) for waves in self.waves_l)
        setup.e_kin_jj = np.zeros((nj, nj))
        setup.id_j = []
        j1 = 0
        for l, waves in enumerate(self.waves_l):
            ne = 0
            for n, f, e, phi_g, phit_g, pt_g in zip(waves.n_n, waves.f_n,
                                                    waves.e_n, waves.phi_ng,
                                                    waves.phit_ng,
                                                    waves.pt_ng):
                setup.append(n, l, f, e, waves.rcut, phi_g, phit_g, pt_g)
                if n == -1:
                    ne += 1
                    id = '%s-%s%d' % (aea.symbol, 'spdf'[l], ne)
                else:
                    id = '%s-%d%s' % (aea.symbol, n, 'spdf'[l])
                setup.id_j.append(id)
            j2 = j1 + len(waves)
            setup.e_kin_jj[j1:j2, j1:j2] = waves.dekin_nn
            j1 = j2

        setup.nc_g = self.nc_g * sqrt(4 * pi)
        setup.nct_g = self.nct_g * sqrt(4 * pi)
        setup.e_kinetic_core = self.ekincore
        setup.vbar_g = self.v0r_g * sqrt(4 * pi)
        setup.vbar_g[1:] /= self.rgd.r_g[1:]
        setup.vbar_g[0] = setup.vbar_g[1]
        setup.Z = aea.Z
        setup.Nc = self.ncore
        setup.Nv = self.nvalence
        setup.e_kinetic = aea.ekin
        setup.e_xc = aea.exc
        setup.e_electrostatic = aea.eH + aea.eZ
        setup.e_total = aea.exc + aea.ekin + aea.eH + aea.eZ
        setup.rgd = self.rgd
        setup.rcgauss = 1 / sqrt(self.alpha)

        self.calculate_exx_integrals()
        setup.ExxC = self.exxcc
        setup.X_p = pack2(self.exxcv_ii)

        setup.tauc_g = self.rgd.zeros()
        setup.tauct_g = self.rgd.zeros()
        #print 'no tau!!!!!!!!!!!'

        if self.aea.scalar_relativistic:
            reltype = 'scalar-relativistic'
        else:
            reltype = 'non-relativistic'
        attrs = [('type', reltype),
                 ('version', 2),
                 ('name', 'gpaw-%s' % version)]
        setup.generatorattrs = attrs

        setup.l0 = self.l0
        setup.e0 = self.e0
        setup.r0 = self.r0
        setup.nderiv0 = self.nderiv0

        return setup

    def calculate_exx_integrals(self):
        # Find core states:
        core = []
        lmax = 0
        for l, ch in enumerate(self.aea.channels):
            for n, phi_g in enumerate(ch.phi_ng):
                if (l >= len(self.waves_l) or
                    (l < len(self.waves_l) and
                     n + l + 1 not in self.waves_l[l].n_n)):
                    core.append((l, phi_g))
                    if l > lmax:
                        lmax = l

        lmax = max(lmax, len(self.waves_l) - 1)
        G_LLL = make_gaunt(lmax)

        # Calculate core contribution to EXX energy:
        self.exxcc = 0.0
        j1 = 0
        for l1, phi1_g in core:
            f = 1.0
            for l2, phi2_g in core[j1:]:
                n_g = phi1_g * phi2_g
                for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                    G = (G_LLL[l1**2:(l1 + 1)**2,
                               l2**2:(l2 + 1)**2,
                               l**2:(l + 1)**2]**2).sum()
                    vr_g = self.rgd.poisson(n_g, l)
                    e = f * self.rgd.integrate(vr_g * n_g, -1) / 4 / pi
                    self.exxcc -= e * G
                f = 2.0
            j1 += 1

        self.log('EXX (core-core):', self.exxcc, 'Hartree')

        # Calculate core-valence contribution to EXX energy:
        nj = sum(len(waves) for waves in self.waves_l)
        ni = sum(len(waves) * (2 * l + 1)
                 for l, waves in enumerate(self.waves_l))

        self.exxcv_ii = np.zeros((ni, ni))

        i1 = 0
        for l1, waves1 in enumerate(self.waves_l):
            for phi1_g in waves1.phi_ng:
                i2 = 0
                for l2, waves2 in enumerate(self.waves_l):
                    for phi2_g in waves2.phi_ng:
                        X_mm = self.exxcv_ii[i1:i1 + 2 * l1 + 1,
                                             i2:i2 + 2 * l2 + 1]
                        if (l1 + l2) % 2 == 0:
                            for lc, phi_g in core:
                                n_g = phi1_g * phi_g
                                for l in range((l1 + lc) % 2,
                                               max(l1, l2) + lc + 1, 2):
                                    vr_g = self.rgd.poisson(phi2_g * phi_g, l)
                                    e = (self.rgd.integrate(vr_g * n_g, -1) /
                                         (4 * pi))
                                    for mc in range(2 * lc + 1):
                                        for m in range(2 * l + 1):
                                            G_L = G_LLL[:,
                                                lc**2 + mc,
                                                l**2 + m]
                                            X_mm += np.outer(
                                                G_L[l1**2:(l1 + 1)**2],
                                                G_L[l2**2:(l2 + 1)**2]) * e
                        i2 += 2 * l2 + 1
                i1 += 2 * l1 + 1


def str2z(x):
    if isinstance(x, int):
        return x
    if x[0].isdigit():
        return int(x)
    return atomic_numbers[x]


def generate(argv=None):
    from optparse import OptionParser

    parser = OptionParser(usage='%prog [options] element',
                          version='%prog 0.1')
    parser.add_option('-f', '--xc-functional', type='string', default='LDA',
                      help='Exchange-Correlation functional ' +
                      '(default value LDA)',
                      metavar='<XC>')
    parser.add_option('-P', '--projectors',
                      help='Projector functions - use comma-separated - ' +
                      'nl values, where n can be pricipal quantum number ' +
                      '(integer) or energy (floating point number). ' +
                      'Example: 2s,0.5s,2p,0.5p,0.0d.')
    parser.add_option('-r', '--radius',
                      help='1.2 or 1.2,1.1,1.1')
    parser.add_option('-0', '--zero-potential',
                      metavar='type,nderivs,radius,e0',
                      help='Parameters for zero potential.')
    parser.add_option('-c', '--pseudo-core-density-radius', type=float,
                      metavar='radius',
                      help='Radius for pseudizing core density.')
    parser.add_option('-z', '--pseudize',
                      metavar='type,nderivs',
                      help='Parameters for pseudizing wave functions.')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-l', '--logarithmic-derivatives',
                      metavar='spdfg,e1:e2:de,radius',
                      help='Plot logarithmic derivatives. ' +
                      'Example: -l spdf,-1:1:0.05,1.3. ' +
                      'Energy range and/or radius can be left out.')
    parser.add_option('-w', '--write', action='store_true')
    parser.add_option('-s', '--scalar-relativistic', action='store_true')
    parser.add_option('--no-check', action='store_true')
    parser.add_option('-t', '--tag', type='string')
    parser.add_option('-a', '--alpha', type=float)

    opt, args = parser.parse_args(argv)

    if len(args) == 0:
        symbols = [symbol for symbol in chemical_symbols
                   if symbol in parameters]
    elif len(args) == 1 and '-' in args[0]:
        Z1, Z2 = args[0].split('-')
        Z1 = str2z(Z1)
        if Z2:
            Z2 = str2z(Z2)
        else:
            Z2 = 86
        symbols = range(Z1, Z2 + 1)
    else:
        symbols = args

    for symbol in symbols:
        Z = str2z(symbol)
        symbol = chemical_symbols[Z]

        kwargs = get_parameters(symbol, opt)
        print kwargs
        gen = _generate(**kwargs)

        if opt.no_check:
            ok = True
        else:
            ok = gen.check_all()

        #gen.test_convergence()

        if opt.write or opt.tag:
            gen.make_paw_setup(opt.tag).write_xml()

        if opt.logarithmic_derivatives or opt.plot:
            import matplotlib.pyplot as plt
            if opt.logarithmic_derivatives:
                r = 1.1 * gen.rcmax
                emin = min(min(wave.e_n) for wave in gen.waves_l) - 0.8
                emax = max(max(wave.e_n) for wave in gen.waves_l) + 0.8
                lvalues, energies, r = parse_ld_str(opt.logarithmic_derivatives,
                                                    (emin, emax, 0.05), r)
                ldmax = 0.0
                for l in lvalues:
                    ld = gen.aea.logarithmic_derivative(l, energies, r)
                    plt.plot(energies, ld, colors[l], label='spdfg'[l])
                    ld = gen.logarithmic_derivative(l, energies, r)
                    plt.plot(energies, ld, '--' + colors[l])

                    # Fixed points:
                    if l < len(gen.waves_l):
                        efix = gen.waves_l[l].e_n
                        ldfix = gen.logarithmic_derivative(l, efix, r)
                        plt.plot(efix, ldfix, 'x' + colors[l])
                        ldmax = max(ldmax, max(abs(ld) for ld in ldfix))

                    if l == gen.l0:
                        efix = [gen.e0]
                        ldfix = gen.logarithmic_derivative(l, efix, r)
                        plt.plot(efix, ldfix, 'x' + colors[l])
                        ldmax = max(ldmax, max(abs(ld) for ld in ldfix))


                if ldmax != 0.0:
                    plt.axis(ymin=-3 * ldmax, ymax=3 * ldmax)
                plt.xlabel('energy [Ha]')
                plt.ylabel(r'$d\phi_{\ell\epsilon}(r)/dr/\phi_{\ell\epsilon}' +
                           r'(r)|_{r=r_c}$')
                plt.legend(loc='best')


            if opt.plot:
                gen.plot()

            try:
                plt.show()
            except KeyboardInterrupt:
                pass
    return gen


def get_parameters(symbol, opt):
    if symbol in parameters:
        projectors, radii, extra = parameters[symbol]
    else:
        projectors, radii, extra = None, 1.0, {}

    if opt.projectors:
        projectors = opt.projectors

    if opt.radius:
        radii = [float(r) for r in opt.radius.split(',')]

    if isinstance(radii, float):
        radii = [radii]

    scale = 1.0#0.9
    radii = [scale * r for r in radii]

    if opt.pseudize:
        type, nderiv = opt.pseudize.split(',')
        pseudize = (type, int(nderiv))
    else:
        pseudize = ('poly', 4)

    l0 = None
    if opt.zero_potential:
        x = opt.zero_potential.split(',')
        type = x[0]
        if len(x) == 1:
            # select only zero_potential type (with defaults)
            # i.e. on the command line: -0 {f,poly}
            nderiv0 = 6
            r0 = max(radii)
        elif len(x) == 2:
            # select zero_potential type, nderivs
            # i.e. on the command line: -0 f,nderivs
            nderiv0 = int(x[1])
            r0 = max(radii)
        else:
            if x[2] in ['min', 'max']:
                # select zero_potential type, nderivs, min/max
                # i.e. on the command line: -0 f,nderivs,{min,max}
                nderiv0 = int(x[1])
                r0 = eval(x[2] + '(radii)')
            else:
                nderiv0 = int(x[1])
                r0 = float(x[2])
        if len(x) == 4:
            e0 = float(x[3])
        elif type == 'poly':
            e0 = None
        else:
            e0 = 0.0

        if type != 'poly':
            l0 = 'spdfg'.find(type)
    else:
        if 'local' not in extra:
            nderiv0 = 2
            #nderiv0 = 3
            e0 = None
            r0 = extra.get('r0', min(radii) / scale) * scale
        else:
            nderiv0 = 5
            #nderiv0 = 6
            r0 = extra.get('r0', min(radii) * 0.9 / scale) * scale
            l0 = 'spdfg'.find(extra['local'])
            e0 = 0.0

    return dict(symbol=symbol,
                xc=opt.xc_functional,
                projectors=projectors,
                radii=radii,
                scalar_relativistic=opt.scalar_relativistic, alpha=opt.alpha,
                l0=l0, r0=r0, nderiv0=nderiv0, e0=e0,
                pseudize=pseudize, rcore=opt.pseudo_core_density_radius)


def _generate(symbol, xc, projectors, radii,
              scalar_relativistic, alpha,
              l0, r0, nderiv0, e0,
              pseudize, rcore):
    aea = AllElectronAtom(symbol, xc)
    gen = PAWSetupGenerator(aea, projectors,
                            scalar_relativistic)#, fd=None)

    gen.construct_shape_function(alpha, radii, eps=1e-10)
    gen.calculate_core_density()
    gen.find_local_potential(l0, r0, nderiv0, e0)
    gen.add_waves(radii)
    gen.pseudize(pseudize[0], pseudize[1], rcore=rcore)
    gen.construct_projectors()
    return gen


if __name__ == '__main__':
    generate()
