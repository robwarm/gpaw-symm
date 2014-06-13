# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as np

from gpaw import debug
import gpaw.mpi as mpi
import _gpaw


class Symmetry:
    
    def __init__(self, id_a, cell_cv, pbc_c=np.ones(3, bool), tolerance=1e-11):
        """Symmetry object.

        Two atoms can only be identical if they have the same atomic numbers,
        setup types and magnetic moments. If it is an LCAO type of calculation,
        they must have the same atomic basis set also.

        """

        self.id_a = id_a
        self.cell_cv = np.array(cell_cv, float)
        assert self.cell_cv.shape == (3, 3)
        self.pbc_c = np.array(pbc_c, bool)
        self.tol = tolerance

        self.op_scc = np.identity(3, int).reshape((1, 3, 3))
        self.a_sa = np.arange(len(id_a)).reshape((1, -1))
        self.inversion = False

    def analyze(self, spos_ac):
        """Determine list of symmetry operations.

        First determine all symmetry operations of the cell. Then call
        ``prune_symmetries`` to remove those symmetries that are not satisfied
        by the atoms.

        """

        # Symmetry operations as matrices in 123 basis
        self.op_scc = [] 
        
        # Metric tensor
        metric_cc = np.dot(self.cell_cv, self.cell_cv.T)

        # Generate all possible 3x3 symmetry matrices using base-3 integers
        power = (6561, 2187, 729, 243, 81, 27, 9, 3, 1)

        # operation is a 3x3 matrix, with possible elements -1, 0, 1, thus
        # there are 3**9 = 19683 possible matrices
        for base3id in xrange(19683):
            op_cc = np.empty((3, 3), dtype=int)
            m = base3id
            for ip, p in enumerate(power):
                d, m = divmod(m, p)
                op_cc[ip // 3, ip % 3] = 1 - d

            # The metric of the cell should be conserved after applying
            # the operation
            opmetric_cc = np.dot(np.dot(op_cc, metric_cc), op_cc.T)
                                       
            if np.abs(metric_cc - opmetric_cc).sum() > self.tol:
                continue

            # Operation must not swap axes that are not both periodic
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if op_cc[~(pbc_cc | np.identity(3, bool))].any():
                continue

            # Operation must not invert axes that are not periodic
            pbc_cc = np.logical_and.outer(self.pbc_c, self.pbc_c)
            if not (op_cc[np.diag(~self.pbc_c)] == 1).all():
                continue

            # operation is a valid symmetry of the unit cell
            self.op_scc.append(op_cc)

        self.op_scc = np.array(self.op_scc)
        
        # Check if symmetry operations are also valid when taking account
        # of atomic positions
        self.prune_symmetries(spos_ac)
        
    def prune_symmetries(self, spos_ac):
        """Remove symmetries that are not satisfied by the atoms."""

        # Build lists of atom numbers for each type of atom - one
        # list for each combination of atomic number, setup type,
        # magnetic moment and basis set
        a_ib = {}
        for a, id in enumerate(self.id_a):
            if id in a_ib:
                a_ib[id].append(a)
            else:
                a_ib[id] = [a]

        opok = []
        a_sa = []
        # Reduce point group using operation matrices
        for op_cc in self.op_scc:
            a_a = np.zeros(len(spos_ac), int)
            ok = True
            for a_b in a_ib.values():
                spos_bc = spos_ac[a_b]
                for a in a_b:
                    spos_c = np.dot(spos_ac[a], op_cc)
                    sdiff_bc = spos_c - spos_bc
                    sdiff_bc -= np.floor(sdiff_bc + 0.5)
                    indices = np.where((sdiff_bc**2).sum(1) < self.tol)[0]
                    if len(indices) == 1:
                        ok = True
                        b = indices[0]
                        a_a[a] = a_b[b]
                    else:
                        assert len(indices) == 0
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                opok.append(op_cc)
                a_sa.append(a_a)

        if debug:
            for op_cc, map_a in zip(opok, a_sa):
                for a1, id1 in enumerate(self.id_a):
                    a2 = map_a[a1]
                    assert id1 == self.id_a[a2]
                    spos1_c = spos_ac[a1]
                    spos2_c = spos_ac[a2]
                    sdiff = np.dot(spos1_c, op_cc) - spos2_c
                    sdiff -= np.floor(sdiff + 0.5)
                    assert np.dot(sdiff, sdiff) < self.tol

        self.a_sa = np.array(a_sa)
        self.op_scc = np.array(opok)
        self.inversion = (self.op_scc ==
                          -np.eye(3, dtype=int)).all(2).all(1).any()

    def check(self, spos_ac):
        """Check if positions satisfy symmetry operations."""

        nsymold = len(self.op_scc)
        self.prune_symmetries(spos_ac)
        if len(self.op_scc) < nsymold:
            raise RuntimeError('Broken symmetry!')

    def reduce(self, bzk_kc, comm=None):
        """Reduce k-points to irreducible part of the BZ.

        Returns the irreducible k-points and the weights and other stuff.
        
        """
        nbzkpts = len(bzk_kc)
        U_scc = self.op_scc
        nsym = len(U_scc)
        
        bz2bz_ks = map_k_points(bzk_kc, U_scc, self.inversion, comm, self.tol)

        bz2bz_k = -np.ones(nbzkpts + 1, int)
        ibz2bz_k = []
        for k in range(nbzkpts - 1, -1, -1):
            # Reverse order looks more natural
            if bz2bz_k[k] == -1:
                bz2bz_k[bz2bz_ks[k]] = k
                ibz2bz_k.append(k)
        ibz2bz_k = np.array(ibz2bz_k[::-1])
        bz2bz_k = bz2bz_k[:-1].copy()

        bz2ibz_k = np.empty(nbzkpts, int)
        bz2ibz_k[ibz2bz_k] = np.arange(len(ibz2bz_k))
        bz2ibz_k = bz2ibz_k[bz2bz_k]

        weight_k = np.bincount(bz2ibz_k) * (1.0 / nbzkpts)

        # Symmetry operation mapping IBZ to BZ:
        sym_k = np.empty(nbzkpts, int)
        for k in range(nbzkpts):
            # We pick the first one found:
            sym_k[k] = np.where(bz2bz_ks[bz2bz_k[k]] == k)[0][0]
        
        # Time-reversal symmetry used on top of the point group operation:
        if self.inversion:
            time_reversal_k = np.zeros(nbzkpts, bool)
        else:
            time_reversal_k = sym_k >= nsym
            sym_k %= nsym

        assert (ibz2bz_k[bz2ibz_k] == bz2bz_k).all()
        for k in range(nbzkpts):
            sign = 1 - 2 * time_reversal_k[k]
            dq_c = (np.dot(U_scc[sym_k[k]], bzk_kc[bz2bz_k[k]]) -
                    sign * bzk_kc[k])
            dq_c -= dq_c.round()
            assert abs(dq_c).max() < 1e-10

        return (bzk_kc[ibz2bz_k], weight_k,
                sym_k, time_reversal_k, bz2ibz_k, ibz2bz_k, bz2bz_ks)

    def prune_symmetries_grid(self, N_c):
        """Remove symmetries that are not satisfied by the grid."""

        U_scc = []
        a_sa = []
        for U_cc, a_a in zip(self.op_scc, self.a_sa):
            if not (U_cc * N_c - (U_cc.T * N_c).T).any():
                U_scc.append(U_cc)
                a_sa.append(a_a)
                
        self.a_sa = np.array(a_sa)
        self.op_scc = np.array(U_scc)

    def symmetrize(self, a, gd):
        """Symmetrize array."""
        
        gd.symmetrize(a, self.op_scc)

    def symmetrize_wavefunction(self, a_g, kibz_c, kbz_c, op_cc,
                                time_reversal):
        """Generate Bloch function from symmetry related function in the IBZ.

        a_g: ndarray
            Array with Bloch function from the irreducible BZ.
        kibz_c: ndarray
            Corresponing k-point coordinates.
        kbz_c: ndarray
            K-point coordinates of the symmetry related k-point.
        op_cc: ndarray
            Point group operation connecting the two k-points.
        time-reversal: bool
            Time-reversal symmetry required in addition to the point group
            symmetry to connect the two k-points.
        
        """

        # Identity
        if (np.abs(op_cc - np.eye(3, dtype=int)) < 1e-10).all():
            if time_reversal:
                return a_g.conj()
            else:
                return a_g
        # Inversion symmetry
        elif (np.abs(op_cc + np.eye(3, dtype=int)) < 1e-10).all():
            return a_g.conj()
        # General point group symmetry
        else:
            import _gpaw
            b_g = np.zeros_like(a_g)
            if time_reversal:
                # assert abs(np.dot(op_cc, kibz_c) - -kbz_c) < tol
                _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc.T.copy(),
                                              kibz_c, -kbz_c)
                return b_g.conj()
            else:
                # assert abs(np.dot(op_cc, kibz_c) - kbz_c) < tol
                _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc.T.copy(),
                                              kibz_c, kbz_c)
                return b_g
        
    def symmetrize_forces(self, F0_av):
        """Symmetrice forces."""
        
        F_ac = np.zeros_like(F0_av)
        for map_a, op_cc in zip(self.a_sa, self.op_scc):
            op_vv = np.dot(np.linalg.inv(self.cell_cv),
                           np.dot(op_cc, self.cell_cv))
            for a1, a2 in enumerate(map_a):
                F_ac[a2] += np.dot(F0_av[a1], op_vv)
        return F_ac / len(self.op_scc)
        
    def print_symmetries(self, text):
        
        n = len(self.op_scc)
        text('Symmetries present: %s' % n)


def map_k_points(bzk_kc, U_scc, inversion, comm=None, tol=1e-11):
    """Find symmetry relations between k-points.

    This is a Python-wrapper for a C-function that does the hard work
    which is distributed over comm.

    The map bz2bz_ks is returned.  If there is a k2 for which::

      = _    _    _
      U q  = q  + N,
       s k1   k2

    where N is a vector of integers, then bz2bz_ks[k1, s] = k2, otherwise
    if there is a k2 for which::

      = _     _    _
      U q  = -q  + N,
       s k1    k2

    then bz2bz_ks[k1, s + nsym] = k2, where nsym = len(U_scc).  Otherwise
    bz2bz_ks[k1, s] = -1.
    """

    if comm is None or isinstance(comm, mpi.DryRunCommunicator):
        comm = mpi.serial_comm

    nbzkpts = len(bzk_kc)
    ka = nbzkpts * comm.rank // comm.size
    kb = nbzkpts * (comm.rank + 1) // comm.size
    assert comm.sum(kb - ka) == nbzkpts

    if not inversion:
        U_scc = np.concatenate([U_scc, -U_scc])

    bz2bz_ks = np.zeros((nbzkpts, len(U_scc)), int)
    bz2bz_ks[ka:kb] = -1
    _gpaw.map_k_points(np.ascontiguousarray(bzk_kc),
                       np.ascontiguousarray(U_scc), tol, bz2bz_ks, ka, kb)
    comm.sum(bz2bz_ks)
    return bz2bz_ks
