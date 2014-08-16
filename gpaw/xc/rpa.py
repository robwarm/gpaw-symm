from __future__ import print_function

import os
import sys
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils import devnull
from scipy.special.orthogonal import p_roots

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


class RPACorrelation:
    def __init__(self, calc, xc='RPA', filename=None,
                 skip_gamma=False, qsym=True, nlambda=None,
                 nfrequencies=16, frequency_max=800.0, frequency_scale=2.0,
                 frequencies=None, weights=None,
                 wcomm=None, chicomm=None, world=mpi.world,
                 txt=sys.stdout):

        if isinstance(calc, str):
            calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
        self.calc = calc

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        if frequencies is None:
            frequencies, weights = get_gauss_legendre_points(nfrequencies,
                                                             frequency_max,
                                                             frequency_scale)
            user_spec = False
        else:
            assert weights is not None
            user_spec = True
            
        self.omega_w = frequencies / Hartree
        self.weight_w = weights / Hartree

        if wcomm is None:
            wcomm = 1

        if isinstance(wcomm, int):
            if wcomm == 1:
                wcomm = mpi.serial_comm
                chicomm = world
            else:
                r = world.rank
                s = world.size
                assert s % wcomm == 0
                n = s // wcomm  # size of skncomm
                wcomm = world.new_communicator(range(r % n, s, n))
                chicomm = world.new_communicator(range(r // n * n,
                                                       (r // n + 1) * n))

        assert len(self.omega_w) % wcomm.size == 0
        self.mynw = len(self.omega_w) // wcomm.size
        self.w1 = wcomm.rank * self.mynw
        self.w2 = self.w1 + self.mynw
        self.myomega_w = self.omega_w[self.w1:self.w2]
        self.wcomm = wcomm
        self.chicomm = chicomm
        self.world = world

        self.skip_gamma = skip_gamma
        self.ibzq_qc = None
        self.weight_q = None
        self.initialize_q_points(qsym)

        # Energies for all q-vetors and cutoff energies:
        self.energy_qi = []

        self.filename = filename

        self.print_initialization(xc, frequency_scale, nlambda, user_spec)

    def initialize_q_points(self, qsym):
        kd = self.calc.wfs.kd
        self.bzq_qc = kd.get_bz_q_points(first=True)

        if not qsym:
            self.ibzq_qc = self.bzq_qc
            self.weight_q = np.ones(len(self.bzq_qc)) / len(self.bzq_qc)
        else:
            U_scc = kd.symmetry.op_scc
            self.ibzq_qc = kd.get_ibz_q_points(self.bzq_qc, U_scc)[0]
            self.weight_q = kd.q_weights

    def read(self):
        lines = open(self.filename).readlines()[1:]
        n = 0
        self.energy_qi = []
        nq = len(lines) // len(self.ecut_i)
        for q_c in self.ibzq_qc[:nq]:
            self.energy_qi.append([])
            for ecut in self.ecut_i:
                q1, q2, q3, ec, energy = [float(x)
                                          for x in lines[n].split()]
                self.energy_qi[-1].append(energy / Hartree)
                n += 1

                if (abs(q_c - (q1, q2, q3)).max() > 1e-4 or
                               abs(int(ecut * Hartree) - ec) > 0):
                    self.energy_qi = []
                    return

        print('Read %d q-points from file: %s' % (nq, self.filename),
              file=self.fd)
        print(file=self.fd)

    def write(self):
        if self.world.rank == 0 and self.filename:
            fd = open(self.filename, 'w')
            print('#%9s %10s %10s %8s %12s' %
                  ('q1', 'q2', 'q3', 'E_cut', 'E_c(q)'), file=fd)
            for energy_i, q_c in zip(self.energy_qi, self.ibzq_qc):
                for energy, ecut in zip(energy_i, self.ecut_i):
                    print('%10.4f %10.4f %10.4f %8d   %r' %
                          (tuple(q_c) + (ecut * Hartree, energy * Hartree)),
                          file=fd)

    def calculate(self, ecut, nbands=None, spin=0):
        """Calculate RPA correlation energy for one or several cutoffs.

        ecut: float or list of floats
            Plane-wave cutoff(s).
        nbands: int
            Number of bands (defaults to number of plane-waves).
        spin: separate spin in response funtion.
            (Only needed for beyond RPA methods that inherit this function)
        """

        if isinstance(ecut, (float, int)):
            ecut_i = [ecut]
            for i in range(5):
                ecut_i.append(ecut_i[-1] * 0.8)
            ecut_i = np.sort(ecut_i)
        else:
            ecut_i = np.sort(ecut)
        self.ecut_i = np.asarray(ecut_i) / Hartree
        ecutmax = max(self.ecut_i)

        if nbands is None:
            print('Response function bands : Equal to number of plane waves',
                  file=self.fd)
        else:
            print('Response function bands : %s' % nbands, file=self.fd)
        print('Plane wave cutoffs (eV) :', end='', file=self.fd)
        for ecut in ecut_i:
            print('%5d' % ecut, end='', file=self.fd)
        print(file=self.fd)
        print(file=self.fd)

        if self.filename and os.path.isfile(self.filename):
            self.read()
            self.world.barrier()

        chi0 = Chi0(self.calc, 1j * Hartree * self.myomega_w, eta=0.0,
                    intraband=False, hilbert=False,
                    txt='response.txt', world=self.chicomm)

        nq = len(self.energy_qi)
        for q_c in self.ibzq_qc[nq:]:
            if np.allclose(q_c, 0.0) and self.skip_gamma:
                self.energy_qi.append(len(self.ecut_i) * [0.0])
                self.write()
                print('Not calculating E_c(q) at Gamma', file=self.fd)
                print(file=self.fd)
                continue

            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(ecutmax, self.calc.wfs.gd, complex, thisqd)
            nG = pd.ngmax

            chi0_swGG = np.zeros((1 + spin, self.mynw, nG, nG), complex)
            if np.allclose(q_c, 0.0):
                # Wings (x=0,1) and head (G=0) for optical limit and three
                # directions (v=0,1,2):
                chi0_swxvG = np.zeros((1 + spin, self.mynw, 2, 3, nG), complex)
                chi0_swvv = np.zeros((1 + spin, self.mynw, 3, 3), complex)
            else:
                chi0_swxvG = None
                chi0_swvv = None

            Q_aGii = chi0.initialize_paw_corrections(pd)

            # First not completely filled band:
            m1 = chi0.nocc1
            print('# %s  -  %s' % (len(self.energy_qi), ctime().split()[-2]),
                  file=self.fd)
            print('q = [%1.3f %1.3f %1.3f]' % tuple(q_c), file=self.fd)

            energy_i = []
            for ecut in self.ecut_i:
                if ecut == ecutmax:
                    # Nothing to cut away:
                    cut_G = None
                    m2 = nbands or nG
                else:
                    cut_G = np.arange(nG)[pd.G2_qG[0] <= 2 * ecut]
                    m2 = len(cut_G)

                print('E_cut = %d eV / Bands = %d:   ' % (ecut * Hartree, m2),
                      file=self.fd, end='')
                self.fd.flush()

                energy = self.calculate_q(chi0, pd,
                                          chi0_swGG, chi0_swxvG, chi0_swvv,
                                          Q_aGii, m1, m2, cut_G)
                energy_i.append(energy)
                m1 = m2

                if ecut < ecutmax and self.chicomm.size > 1:
                    # Chi0 will be summed again over chicomm, so we divide
                    # by its size:
                    chi0_swGG *= 1.0 / self.chicomm.size
                    if chi0_swxvG is not None:
                        chi0_swxvG *= 1.0 / self.chicomm.size
                        chi0_swvv *= 1.0 / self.chicomm.size

            self.energy_qi.append(energy_i)
            self.write()
            print(file=self.fd)

        e_i = np.dot(self.weight_q, np.array(self.energy_qi))
        print('==========================================================',
              file=self.fd)
        print(file=self.fd)
        print('Total correlation energy:', file=self.fd)
        for e_cut, e in zip(self.ecut_i, e_i):
            print('%6.0f:   %6.4f eV' % (e_cut * Hartree, e * Hartree),
                  file=self.fd)
        print(file=self.fd)

        self.energy_qi = []  # important if another calculation is performed

        if len(e_i) > 1:
            self.extrapolate(e_i)

        print('Calculation completed at: ', ctime(), file=self.fd)
        print(file=self.fd)

        return e_i * Hartree

    def calculate_q(self, chi0, pd,
                    chi0_swGG, chi0_swxvG, chi0_swvv, Q_aGii, m1, m2, cut_G):
        chi0_wGG = chi0_swGG[0]
        if chi0_swxvG is not None:
            chi0_wxvG = chi0_swxvG[0]
            chi0_wvv = chi0_swvv[0]
        else:
            chi0_wxvG = None
            chi0_wvv = None
        chi0._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv,
                        Q_aGii, m1, m2, [0, 1])

        print('E_c(q) = ', end='', file=self.fd)

        if not pd.kd.gamma:
            e = self.calculate_energy(pd, chi0_wGG, cut_G)
            print('%.3f eV' % (e * Hartree), file=self.fd)
            self.fd.flush()
        else:
            e = 0.0
            for v in range(3):
                chi0_wGG[:, 0] = chi0_wxvG[:, 0, v]
                chi0_wGG[:, :, 0] = chi0_wxvG[:, 1, v]
                chi0_wGG[:, 0, 0] = chi0_wvv[:, v, v]
                ev = self.calculate_energy(pd, chi0_wGG, cut_G)
                e += ev
                print('%.3f' % (ev * Hartree), end='', file=self.fd)
                if v < 2:
                    print('/', end='', file=self.fd)
                else:
                    print(' eV', file=self.fd)
                    self.fd.flush()
            e /= 3

        return e

    def calculate_energy(self, pd, chi0_wGG, cut_G):
        """Evaluate correlation energy from chi0."""

        G_G = pd.G2_qG[0]**0.5  # |G+q|
        if pd.kd.gamma:
            G_G[0] = 1.0

        if cut_G is not None:
            G_G = G_G[cut_G]

        nG = len(G_G)

        e_w = []
        for chi0_GG in chi0_wGG:
            if cut_G is not None:
                chi0_GG = chi0_GG.take(cut_G, 0).take(cut_G, 1)

            e_GG = np.eye(nG) - 4 * np.pi * chi0_GG / G_G / G_G[:, np.newaxis]
            e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
            e_w.append(e.real)

        E_w = np.zeros_like(self.omega_w)
        self.wcomm.all_gather(np.array(e_w), E_w)
        energy = np.dot(E_w, self.weight_w) / (2 * np.pi)
        self.E_w = E_w
        return energy

    def extrapolate(self, e_i):
        print('Extrapolated energies:', file=self.fd)
        ex_i = []
        for i in range(len(e_i) - 1):
            e1, e2 = e_i[i:i + 2]
            x1, x2 = self.ecut_i[i:i + 2]**-1.5
            ex = (e1 * x2 - e2 * x1) / (x2 - x1)
            ex_i.append(ex)

            print('  %4.0f -%4.0f:  %5.3f eV' % (self.ecut_i[i] * Hartree,
                                                 self.ecut_i[i + 1] * Hartree,
                                                 ex * Hartree),
                  file=self.fd)
        print(file=self.fd)
        self.fd.flush()

        return e_i * Hartree

    def print_initialization(self, xc, frequency_scale, nlambda, user_spec):
        print('----------------------------------------------------------',
              file=self.fd)
        print('Non-self-consistent %s correlation energy' % xc, file=self.fd)
        print('----------------------------------------------------------',
              file=self.fd)
        print('Started at:  ', ctime(), file=self.fd)
        print(file=self.fd)
        print('Atoms                          :',
              self.calc.atoms.get_chemical_formula(mode='hill'), file=self.fd)
        print('Ground state XC functional     :',
              self.calc.hamiltonian.xc.name, file=self.fd)
        print('Valence electrons              :',
              self.calc.wfs.setups.nvalence, file=self.fd)
        print('Number of bands                :',
              self.calc.wfs.bd.nbands, file=self.fd)
        print('Number of spins                :',
              self.calc.wfs.nspins, file=self.fd)
        print('Number of k-points             :',
              len(self.calc.wfs.kd.bzk_kc), file=self.fd)
        print('Number of irreducible k-points :',
              len(self.calc.wfs.kd.ibzk_kc), file=self.fd)
        print('Number of q-points             :',
              len(self.bzq_qc), file=self.fd)
        print('Number of irreducible q-points :',
              len(self.ibzq_qc), file=self.fd)
        print(file=self.fd)
        for q, weight in zip(self.ibzq_qc, self.weight_q):
            print('    q: [%1.4f %1.4f %1.4f] - weight: %1.3f' %
                  (q[0], q[1], q[2], weight), file=self.fd)
        print(file=self.fd)
        print('----------------------------------------------------------',
              file=self.fd)
        print('----------------------------------------------------------',
              file=self.fd)
        print(file=self.fd)
        if nlambda is None:
            print('Analytical coupling constant integration', file=self.fd)
        else:
            print('Numerical coupling constant integration using', nlambda,
                  'Gauss-Legendre points', file=self.fd)
        print(file=self.fd)
        print('Frequencies', file=self.fd)
        if not user_spec:
            print('    Gauss-Legendre integration with %s frequency points' %
                  len(self.omega_w), file=self.fd)
            print('    Transformed from [0,oo] to [0,1] using e^[-aw^(1/B)]',
                  file=self.fd)
            print('    Highest frequency point at %5.1f eV and B=%1.1f' %
                  (self.omega_w[-1] * Hartree, frequency_scale), file=self.fd)
        else:
            print('    User specified frequency integration with',
                  len(self.omega_w), 'frequency points', file=self.fd)
        print(file=self.fd)
        print('Parallelization', file=self.fd)
        print('    Total number of CPUs          : % s' % self.world.size,
              file=self.fd)
        print('    Frequency decomposition       : % s' % self.wcomm.size,
              file=self.fd)
        print('    K-point/band decomposition    : % s' % self.chicomm.size,
              file=self.fd)
        print(file=self.fd)


def get_gauss_legendre_points(nw=16, frequency_max=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    y_w = y_w.real
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_max / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_max
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    return w, weights_w * transform / 2
