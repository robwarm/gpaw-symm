import sys
from math import pi

import numpy as np
from ase.units import Hartree
from ase.utils import devnull, prnt

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.utilities.blas import gemm
from gpaw.fd_operators import Gradient
from gpaw.wavefunctions.pw import PWLFC
from gpaw.response.math_func import two_phi_planewave_integrals


class KPoint:
    def __init__(self, s, K, n1, n2, ut_nR, eps_n, f_n, P_ani, shift_c):
        self.s = s    # spin index
        self.K = K    # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.ut_nR = ut_nR      # periodic part of wave functions in real-space
        self.eps_n = eps_n      # eigenvalues
        self.f_n = f_n          # occupation numbers
        self.P_ani = P_ani      # PAW projections
        self.shift_c = shift_c  # long story - see the
                                # PairDensity.construct_symmetry_operators()
                                # method


class PairDensity:
    def __init__(self, calc, ecut=50,
                 ftol=1e-6,
                 real_space_derivatives=False,
                 world=mpi.world, txt=sys.stdout):
        ecut /= Hartree
        
        self.ecut = ecut
        self.ftol = ftol
        self.real_space_derivatives = real_space_derivatives
        self.world = world
        
        if isinstance(calc, str):
            calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
        self.calc = calc

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt
        
        self.spos_ac = calc.atoms.get_scaled_positions()
        
        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()
        
        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        self.ut_sKnvR = None  # gradient of wave functions for optical limit

    def count_occupied_bands(self):
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - self.ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > self.ftol).sum(), self.nocc2)
        prnt('Number of completely filled bands:', self.nocc1, file=self.fd)
        prnt('Number of non-empty bands:', self.nocc2, file=self.fd)
        prnt('Total number of bands:', self.calc.wfs.bd.nbands,
             file=self.fd)
        
    def distribute_k_points_and_bands(self, nbands):
        """Distribute spins, k-points and bands.
        
        nbands: int
            Number of bands for each spin/k-point combination.
            
        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """
        
        world = self.world
        wfs = self.calc.wfs
        ns = wfs.nspins
        nk = wfs.kd.nbzkpts
        n = (ns * nk * nbands + world.size - 1) // world.size
        i1 = world.rank * n
        i2 = min((world.rank + 1) * n, ns * nk * nbands)

        self.mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in range(nk):
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    self.mysKn1n2.append((s, K, n1, n2))
                i += nbands

        prnt('BZ k-points:', self.calc.wfs.kd.description, file=self.fd)
        prnt('Distributing spins, k-points and bands (%d x %d x %d)' %
             (ns, nk, nbands),
             'over %d process%s' %
              (world.size, ['es', ''][world.size == 1]), file=self.fd)
            
    def get_k_point(self, s, K, n1, n2):
        """Return wave functions for a specific k-point and spin.
        
        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        """
        wfs = self.calc.wfs
        
        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        
        psit_nG = kpt.psit_nG
        ut_nR = wfs.gd.empty(n2 - n1, wfs.dtype)
        for n in range(n1, n2):
            ut_nR[n - n1] = T(wfs.pd.ifft(psit_nG[n], ik))
            
        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight
        
        P_ani = []
        for b, U_ii in zip(a_a, U_aii):
            P_ni = np.dot(kpt.P_ani[b][n1:n2], U_ii)
            if time_reversal:
                P_ni = P_ni.conj()
            P_ani.append(P_ni)
        
        return KPoint(s, K, n1, n2, ut_nR, eps_n, f_n, P_ani, shift_c)
    
    def calculate_pair_densities(self, ut1cc_R, C1_aGi, kpt2, pd, Q_G):
        """Calculate FFT of pair-denities and add PAW corrections.
        
        ut1cc_R: 3-d complex ndarray
            Complex conjugate of the periodic part of the left hand side
            wave function.
        C1_aGi: list of ndarrays
            PAW corrections for all atoms.
        kpt2: KPoint object
            Right hand side k-point object.
        pd: PWDescriptor
            Plane-wave descriptor for for q=k2-k1.
        Q_G: 1-d int ndarray
            Mapping from flattened 3-d FFT grid to 0.5(G+q)^2<ecut sphere.
        """
        
        dv = pd.gd.dv
        n_mG = pd.empty(kpt2.n2 - kpt2.n1)
        for ut_R, n_G in zip(kpt2.ut_nR, n_mG):
            n_R = ut1cc_R * ut_R
            n_G[:] = pd.fft(n_R, 0, Q_G) * dv
            
        # PAW corrections:
        for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
            gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG, 't')
            
        return n_mG
    
    def get_fft_indices(self, K1, K2, q_c, pd, shift0_c):
        """Get indices for G-vectors inside cutoff sphere."""
        kd = self.calc.wfs.kd
        Q_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K1]).round().astype(int))
        if shift_c.any():
            q_cG = np.unravel_index(Q_G, pd.gd.N_c)
            q_cG = [q_G + shift for q_G, shift in zip(q_cG, shift_c)]
            Q_G = np.ravel_multi_index(q_cG, pd.gd.N_c, 'wrap')
        return Q_G
        
    def construct_symmetry_operators(self, K):
        """Construct symmetry operators for wave function and PAW projections.
        
        We want to transform a k-point in the irreducible part of the BZ to
        the corresponding k-point with index K.
        
        Returns U_cc, T, a_a, U_aii, shift_c and time_reversal, where:
        
        * U_cc is a rotation matrix.
        * T() is a function that transforms the periodic part of the wave
          function.
        * a_a is a list of symmetry related atom indices
        * U_aii is a list of rotation matrices for the PAW projections
        * shift_c is three integers: see code below.
        * time_reversal is a flag - if True, projections should be complex
          conjugated.
            
        See the get_k_point() method for how to use these tuples.
        """
        
        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]
        
        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)
        
        if (U_cc == np.eye(3)).all():
            T = lambda f_R: f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            T = lambda f_R: f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T
            T = lambda f_R: T0(f_R).conj()
            shift_c *= -1
        
        a_a = []
        U_aii = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

    def initialize_paw_corrections(self, pd, soft=False):
        wfs = self.calc.wfs
        q_v = pd.K_qv[0]
        optical_limit = not q_v.any()

        G_Gv = pd.G_Qv[pd.Q_qG[0]] + q_v
        if optical_limit:
            G_Gv[0] = 1
            
        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)
        
        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            if soft:
                ghat = PWLFC([atomdata.ghat_l], pd)
                ghat.set_positions(np.zeros((1, 3)))
                Q_LG = ghat.expand()
                Q_Gii = np.dot(atomdata.Delta_iiL, Q_LG).T
            else:
                Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
                ni = atomdata.ni
                Q_Gii.shape = (-1, ni, ni)
                
            Q_xGii[id] = Q_Gii
            
        Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_Gii = Q_xGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
            if optical_limit:
                Q_aGii[a][0] = atomdata.dO_ii
                
        return Q_aGii

    def update_optical_limit(self, n, kpt1, kpt2, deps_m, df_m, n_mG):
        if self.ut_sKnvR is None:
            self.ut_sKnvR = self.calculate_derivatives()
            
        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n]
        
        atomdata_a = self.calc.wfs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]
        
        n0_mv = -self.calc.wfs.gd.integrate(ut_vR, kpt2.ut_nR).T
        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            gemm(1.0, C_vi, P_mi, 1.0, n0_mv, 'c')

        deps_m = deps_m.copy()
        deps_m[deps_m > -1e-3] = np.inf
        n0_mv *= 1j / deps_m[:, np.newaxis]
        n_mG[:, 0] = n0_mv[:, 0]
        
        return n0_mv

    def calculate_derivatives(self):
        if self.real_space_derivatives:
            grad_v = [Gradient(self.calc.wfs.gd, v, 1.0, 4, complex).apply
                      for v in range(3)]
        wfs = self.calc.wfs
        ut_sKnvR = [{}, {}]
        for s, K, n1, n2 in self.mysKn1n2:
            U_cc, T, a_a, U_aii, shift_c, time_reversal = \
                self.construct_symmetry_operators(K)
            A_cv = wfs.gd.cell_cv
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
            psit_nG = kpt.psit_nG
            iG_Gv = 1j * wfs.pd.G_Qv[wfs.pd.Q_qG[ik]]
            ut_nvR = wfs.gd.zeros((n2 - n1, 3), complex)
            for n in range(n1, n2):
                for v in range(3):
                    if self.real_space_derivatives:
                        ut_R = T(wfs.pd.ifft(psit_nG[n], ik))
                        grad_v[v](ut_R, ut_nvR[n - n1, v],
                                  np.ones((3, 2), complex))
                    else:
                        ut_R = T(wfs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                        for v2 in range(3):
                            ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]
            ut_sKnvR[s][K] = ut_nvR
            
        return ut_sKnvR
