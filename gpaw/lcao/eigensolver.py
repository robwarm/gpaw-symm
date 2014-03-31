import numpy as np

from gpaw.utilities import unpack
from gpaw.utilities.blas import gemm
import gpaw.mpi as mpi


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self, diagonalizer=None):
        self.diagonalizer = diagonalizer
        # ??? why should we be able to set
        # this diagonalizer in both constructor and initialize?
        self.has_initialized = False # XXX

    def initialize(self, gd, dtype, nao, diagonalizer=None):
        self.gd = gd
        self.nao = nao
        if diagonalizer is not None:
            self.diagonalizer = diagonalizer
        assert self.diagonalizer is not None
        self.has_initialized = True # XXX

    def reset(self):
        pass

    def error(self):
        return 0.0
    error = property(error)

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, Vt_xMM=None,
                                     root=-1):
        # XXX document parallel stuff, particularly root parameter
        assert self.has_initialized

        bf = wfs.basis_functions
        
        if Vt_xMM is None:
            wfs.timer.start('Potential matrix')
            vt_G = hamiltonian.vt_sG[kpt.s]
            Vt_xMM = bf.calculate_potential_matrices(vt_G)
            wfs.timer.stop('Potential matrix')

        if bf.gamma:
            y = 1.0
            H_MM = Vt_xMM[0]
            if wfs.dtype == complex:
                H_MM = H_MM.astype(complex)
        else:
            wfs.timer.start('Sum over cells')
            y = 0.5
            k_c = wfs.kd.ibzk_qc[kpt.q]
            H_MM = (0.5 + 0.0j) * Vt_xMM[0]
            for sdisp_c, Vt_MM in zip(bf.sdisp_xc, Vt_xMM)[1:]:
                H_MM += np.exp(2j * np.pi * np.dot(sdisp_c, k_c)) * Vt_MM
            wfs.timer.stop('Sum over cells')
        
        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        wfs.timer.start('Atomic Hamiltonian')
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][kpt.s]), wfs.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), wfs.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            gemm(y, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_MM)
        wfs.timer.stop('Atomic Hamiltonian')
        wfs.timer.start('Distribute overlap matrix')
        H_MM = wfs.ksl.distribute_overlap_matrix(
            H_MM, root, add_hermitian_conjugate=(y == 0.5))
        wfs.timer.stop('Distribute overlap matrix')
        H_MM += wfs.T_qMM[kpt.q]
        return H_MM

    def iterate(self, hamiltonian, wfs):
        wfs.timer.start('LCAO eigensolver')

        s = -1
        for kpt in wfs.kpt_u:
            if kpt.s != s:
                s = kpt.s
                wfs.timer.start('Potential matrix')
                Vt_xMM = wfs.basis_functions.calculate_potential_matrices(
                    hamiltonian.vt_sG[s])
                wfs.timer.stop('Potential matrix')
            self.iterate_one_k_point(hamiltonian, wfs, kpt, Vt_xMM)

        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, hamiltonian, wfs, kpt, Vt_xMM):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, Vt_xMM,
                                                 root=0)
        S_MM = wfs.S_qMM[kpt.q]

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(wfs.bd.mynbands)
            
        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nM, kpt.eps_n, S_MM)
        wfs.timer.stop(diagonalization_string)

        wfs.timer.start('Calculate projections')
        # P_ani are not strictly necessary as required quantities can be
        # evaluated directly using P_aMi.  We should probably get rid
        # of the places in the LCAO code using P_ani directly
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')
        wfs.timer.stop('Calculate projections')

    def estimate_memory(self, mem, dtype):
        pass 
        # self.diagonalizer.estimate_memory(mem, dtype) #XXX enable this
