from __future__ import print_function

import os
import sys
import pickle
from math import pi

import numpy as np
from ase.units import Hartree, Bohr

import gpaw.mpi as mpi
from gpaw.response.chi0 import Chi0
from gpaw.response.kernel2 import calculate_Kxc
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb


class DielectricFunction:
    """This class defines dielectric function related physical quantities."""
    def __init__(self, calc, name=None, frequencies=None, domega0=0.1,
                 omega2=10.0, omegamax=None, ecut=50,
                 hilbert=True, nbands=None,
                 eta=0.2, ftol=1e-6, intraband=True, world=mpi.world,
                 txt=sys.stdout, nthreads=1, gate_voltage=None):

        self.chi0 = Chi0(calc, frequencies, domega0=domega0,
                         omega2=omega2, omegamax=omegamax,
                         ecut=ecut, hilbert=hilbert, nbands=nbands,
                         eta=eta, ftol=ftol,
                         intraband=intraband, nthreads=nthreads,
                         world=world, txt=txt,
                         gate_voltage=gate_voltage)
        
        self.name = name

    def calculate_chi0(self, q_c):
        if self.name:
            kd = self.chi0.calc.wfs.kd
            name = self.name + '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round())
            if os.path.isfile(name):
                try:
                    omega_w, pd, chi0_wGG, chi0_wxvG, chi0_wvv = \
                        pickle.load(open(name))
                    print('Reading from file ', name)
                except EOFError:
                    pass
                else:
                    return pd, chi0_wGG, chi0_wxvG, chi0_wvv

        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.chi0.calculate(q_c)
        self.chi0.timer.write(self.chi0.fd)
        
        if self.name and mpi.rank == 0:
            with open(name, 'wb') as fd:
                pickle.dump((self.chi0.omega_w, pd,
                             chi0_wGG, chi0_wxvG, chi0_wvv), fd,
                            pickle.HIGHEST_PROTOCOL)

        # Wait for rank 0 to save Chi
        mpi.world.barrier()
        
        # Not returning frequencies will work for now
        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    def get_chi(self, xc='RPA', q_c=[0, 0, 0], direction='x',
                wigner_seitz_truncation=False):
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5
        nG = len(G_G)
        
        if pd.kd.gamma:
            G_G[0] = 1.0
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction
            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv, d_v).T)
        
        G_G /= (4 * pi)**0.5

        if wigner_seitz_truncation:
            kernel = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv,
                                                 self.chi0.calc.wfs.kd.N_c)
            K_G = kernel.get_potential(pd)
            K_G *= G_G**2
            if pd.kd.gamma:
                K_G[0] = 0.0
        else:
            K_G = np.ones(nG)

        K_GG = np.zeros((nG, nG), dtype=complex)
        for i in range(nG):
            K_GG[i, i] = K_G[i]

        if xc != 'RPA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG
            K_GG += calculate_Kxc(pd, nt_sG, R_av, self.chi0.calc.wfs.setups,
                                  self.chi0.calc.density.D_asp,
                                  functional=xc) * G_G * G_G[:, np.newaxis]
            
        chi_wGG = []
        for chi0_GG in chi0_wGG:
            chi0_GG[:] = chi0_GG / G_G / G_G[:, np.newaxis]
            chi_wGG.append(np.dot(np.linalg.inv(np.eye(nG) -
                                                np.dot(chi0_GG, K_GG)),
                                  chi0_GG))
        return chi0_wGG, np.array(chi_wGG)
  
    def get_dielectric_matrix(self, xc='RPA', q_c=[0, 0, 0],
                              direction='x', wigner_seitz_truncation=False,
                              symmetric=True):
        """Returns the symmetrized dielectric matrix.
        
        ::
        
            \tilde\epsilon_GG' = v^{-1/2}_G \epsilon_GG' v^{-1/2}_G',
            
        where::
            
            epsilon_GG' = 1 - v_G * P_GG' and P_GG'
            
        is the polarization.
        
        ::
            
            In RPA:   P = chi^0
            In TDDFT: P = (1 - chi^0 * f_xc)^{-1} chi^0
        
        The head of the inverse symmetrized dielectric matrix is equal
        to the head of the inverse dielectric matrix (inverse dielectric
        function)
        """
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        G_G = pd.G2_qG[0]**0.5
        nG = len(G_G)

        if pd.kd.gamma:
            G_G[0] = 1.0
            if isinstance(direction, str):
                d_v = {'x': [1, 0, 0],
                       'y': [0, 1, 0],
                       'z': [0, 0, 1]}[direction]
            else:
                d_v = direction

            d_v = np.asarray(d_v) / np.linalg.norm(d_v)
            chi0_wGG[:, 0] = np.dot(d_v, chi0_wxvG[:, 0])
            chi0_wGG[:, :, 0] = np.dot(d_v, chi0_wxvG[:, 1])
            chi0_wGG[:, 0, 0] = np.dot(d_v, np.dot(chi0_wvv, d_v).T)
                    
        if wigner_seitz_truncation:
            kernel = WignerSeitzTruncatedCoulomb(pd.gd.cell_cv,
                                                 self.chi0.calc.wfs.kd.N_c)
            K_G = kernel.get_potential(pd)**0.5
            if pd.kd.gamma:
                K_G[0] = 0.0
        else:
            K_G = (4 * pi)**0.5 / G_G

        if xc != 'RPA':
            R_av = self.chi0.calc.atoms.positions / Bohr
            nt_sG = self.chi0.calc.density.nt_sG
            Kxc_sGG = calculate_Kxc(pd, nt_sG, R_av,
                                    self.chi0.calc.wfs.setups,
                                    self.chi0.calc.density.D_asp,
                                    functional=xc)

        for chi0_GG in chi0_wGG:
            if xc == 'RPA':
                P_GG = chi0_GG
            else:
                P_GG = np.dot(np.linalg.inv(np.eye(nG) -
                                            np.dot(chi0_GG, Kxc_sGG[0])),
                              chi0_GG)
            if symmetric:
                e_GG = np.eye(nG) - P_GG * K_G * K_G[:, np.newaxis]
            else:
                K_GG = (K_G**2 * np.ones([nG, nG])).T
                e_GG = np.eye(nG) - P_GG * K_GG
            chi0_GG[:] = e_GG

        # chi0_wGG is now the dielectric matrix
        return chi0_wGG

    def get_dielectric_function(self, xc='RPA', q_c=[0, 0, 0],
                                direction='x', filename='df.csv',
                                wigner_seitz_truncation=False):
        """Calculate the dielectric function.

        Returns dielectric function without and with local field correction:
        df_NLFC_w, df_LFC_w = DielectricFunction.get_dielectric_function()
        """
        e_wGG = self.get_dielectric_matrix(xc, q_c, direction,
                                           wigner_seitz_truncation)
        df_NLFC_w = np.zeros(len(e_wGG), dtype=complex)
        df_LFC_w = np.zeros(len(e_wGG), dtype=complex)

        for w, e_GG in enumerate(e_wGG):
            df_NLFC_w[w] = e_GG[0, 0]
            df_LFC_w[w] = 1 / np.linalg.inv(e_GG)[0, 0]
        
        if filename is not None and mpi.rank == 0:
            with open(filename, 'w') as fd:
                for omega, nlfc, lfc in zip(self.chi0.omega_w * Hartree,
                                            df_NLFC_w,
                                            df_LFC_w):
                    print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                          (omega, nlfc.real, nlfc.imag, lfc.real, lfc.imag),
                          file=fd)
                
        return df_NLFC_w, df_LFC_w

    def get_macroscopic_dielectric_constant(self, xc='RPA', direction='x',
                                            wigner_seitz_truncation=False):
        """Calculate macroscopic dielectric constant.
        
        Returns eM_NLFC and eM_LFC.

        Macroscopic dielectric constant is defined as the real part
        of dielectric function at w=0.
        
        Parameters:

        eM_LFC: float
            Dielectric constant without local field correction. (RPA, ALDA)
        eM2_NLFC: float
            Dielectric constant with local field correction.
        """

        wst = wigner_seitz_truncation
        fd = self.chi0.fd
        print('', file=fd)
        print('%s Macroscopic Dielectric Constant:' % xc, file=fd)
       
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(
            xc=xc,
            filename=None,
            direction=direction,
            wigner_seitz_truncation=wst)
        eps0 = np.real(df_NLFC_w[0])
        eps = np.real(df_LFC_w[0])
        print('  %s direction' % direction, file=fd)
        print('    Without local field: %f' % eps0, file=fd)
        print('    Include local field: %f' % eps, file=fd)
            
        return eps0, eps

    def get_eels_spectrum(self, xc='RPA', q_c=[0, 0, 0],
                          direction='x', filename='eels.csv',
                          wigner_seitz_truncation=False):
        """Calculate EELS spectrum. By default, generate a file 'eels.csv'.

        EELS spectrum is obtained from the imaginary part of the inverse
        of dielectric function. Returns EELS spectrum without and with
        local field corrections:

        df_NLFC_w, df_LFC_w = DielectricFunction.get_eels_spectrum()
        """

        # Calculate dielectric function
        df_NLFC_w, df_LFC_w = self.get_dielectric_function(
            xc=xc, q_c=q_c,
            direction=direction,
            filename=None,
            wigner_seitz_truncation=wigner_seitz_truncation)
        Nw = df_NLFC_w.shape[0]
        
        # Calculate eels
        eels_NLFC_w = -(1 / df_NLFC_w).imag
        eels_LFC_w = -(1 / df_LFC_w).imag

        # Write to file
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            print('# energy, eels_NLFC_w, eels_LFC_w', file=fd)
            for iw in range(Nw):
                print('%.6f, %.6f, %.6f' %
                     (self.chi0.omega_w[iw] * Hartree,
                      eels_NLFC_w[iw], eels_LFC_w[iw]), file=fd)
            fd.close()

        return eels_NLFC_w, eels_LFC_w
        
    def get_polarizability(self, xc='RPA', direction='x',
                           wigner_seitz_truncation=False,
                           filename='polarizability.csv', pbc=None):
        """Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.

        By default, generate a file 'polarizability.csv'. The five colomns are:
        frequency (eV), Real(alpha0), Imag(alpha0), Real(alpha), Imag(alpha)
        alpha0 is the result without local field effects and the
        dimension of alpha is \AA to the power of non-periodic directions
        """

        cell_cv = self.chi0.calc.wfs.gd.cell_cv
        if not pbc:
            pbc_c = self.chi0.calc.atoms.pbc
        else:
            pbc_c = np.array(pbc)
        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

        if not wigner_seitz_truncation:
            # Without truncation alpha is simply related to eps_M
            df0_w, df_w = self.get_dielectric_function(xc=xc, q_c=[0, 0, 0],
                                                       filename=None,
                                                       direction=direction)
            alpha_w = V * (df_w - 1.0) / (4 * pi)
            alpha0_w = V * (df0_w - 1.0) / (4 * pi)
        else:
            # With truncation we need to calculate \chit = v^0.5*chi*v^0.5
            print('Using Wigner-Seitz truncated Coulomb interaction',
                  file=self.chi0.fd)
            chi0_wGG, chi_wGG = self.get_chi(xc=xc, direction=direction,
                                             wigner_seitz_truncation=True)
            alpha_w = -V * (chi_wGG[:, 0, 0]) / (4 * pi)
            alpha0_w = -V * (chi0_wGG[:, 0, 0]) / (4 * pi)

        Nw = len(alpha_w)
        if filename is not None and mpi.rank == 0:
            fd = open(filename, 'w')
            for iw in range(Nw):
                print('%.6f, %.6f, %.6f, %.6f, %.6f' %
                      (self.chi0.omega_w[iw] * Hartree,
                       alpha0_w[iw].real * Bohr**(sum(~pbc_c)),
                       alpha0_w[iw].imag * Bohr**(sum(~pbc_c)),
                       alpha_w[iw].real * Bohr**(sum(~pbc_c)),
                       alpha_w[iw].imag * Bohr**(sum(~pbc_c))), file=fd)
            fd.close()

        return alpha0_w * Bohr**(sum(~pbc_c)), alpha_w * Bohr**(sum(~pbc_c))

    def check_sum_rule(self, spectrum=None):
        """Check f-sum rule.
        
        It takes the y of a spectrum as an entry and it check its integral.
        """
        
        fd = self.chi0.fd
        
        if spectrum is None:
            raise ValueError('No spectrum input ')
        dw = self.chi0.omega_w[1] - self.chi0.omega_w[0]
        N1 = 0
        for iw in range(len(spectrum)):
            w = iw * dw
            N1 += spectrum[iw] * w
        N1 *= dw * self.chi0.vol / (2 * pi**2)

        print('', file=fd)
        print('Sum rule:', file=fd)
        nv = self.chi0.calc.wfs.nvalence
        print('N1 = %f, %f  %% error' % (N1, (N1 - nv) / nv * 100), file=fd)

    def get_eigenmodes(self, q_c=[0, 0, 0], w_max=None, name=None,
                       eigenvalue_only=False):
        
        """Plasmon eigenmodes as eigenvectors of the dielectric matrix."""

        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        e_wGG = self.get_dielectric_matrix(xc='RPA', q_c=q_c,
                                           wigner_seitz_truncation=True,
                                           symmetric=False)
        
        kd = self.chi0.calc.wfs.kd
        
        # Get real space grid for plasmon modes:
        r = pd.gd.get_grid_point_coordinates()
        w_w = self.chi0.omega_w * Hartree
        if w_max:
            w_w = w_w[np.where(w_w < w_max)]
        Nw = len(w_w)
        nG = e_wGG.shape[1]
             
        eig = np.zeros([Nw, nG], dtype=complex)
        eig_all = np.zeros([Nw, nG], dtype=complex)
      
        # Find eigenvalues and eigenvectors:
        e_GG = e_wGG[0]
        eig_all[0], vec = np.linalg.eig(e_GG)
        eig[0] = eig_all[0]
        vec_dual = np.linalg.inv(vec)
        omega0 = np.array([])
        eigen0 = np.array([], dtype=complex)
        v_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]],
                         dtype=complex)
        n_ind = np.zeros([0, r.shape[1], r.shape[2], r.shape[3]],
                         dtype=complex)
         
        # Loop to find the eigenvalues that crosses zero
        # from negative to positive values:
        for i in np.array(range(1, Nw)):
            e_GG = e_wGG[i]  # epsilon_GG'(omega + d-omega)
            eig_all[i], vec_p = np.linalg.eig(e_GG)
            if eigenvalue_only:
                continue
            vec_dual_p = np.linalg.inv(vec_p)
            overlap = np.abs(np.dot(vec_dual, vec_p))
            index = list(np.argsort(overlap)[:, -1])
            if len(np.unique(index)) < nG:  # add missing indices
                addlist = []
                removelist = []
                for j in range(nG):
                    if index.count(j) < 1:
                        addlist.append(j)
                    if index.count(j) > 1:
                        for l in range(1, index.count(j)):
                            removelist.append(
                                np.argwhere(np.array(index) == j)[l])
                for j in range(len(addlist)):
                    index[removelist[j]] = addlist[j]
            vec = vec_p[:, index]
            vec_dual = vec_dual_p[index, :]
            eig[i] = eig_all[i, index]
            for k in [k for k in range(nG)
                      # Eigenvalue crossing:
                      if (eig[i - 1, k] < 0 and eig[i, k] > 0)]:
                a = np.real((eig[i, k] - eig[i - 1, k]) /
                            (w_w[i] - w_w[i - 1]))
                # linear interp for crossing point
                w0 = np.real(-eig[i - 1, k]) / a + w_w[i - 1]
                eig0 = a * (w0 - w_w[i - 1]) + eig[i - 1, k]
                print('crossing found at w = %1.2f eV' % w0)
                omega0 = np.append(omega0, w0)
                eigen0 = np.append(eigen0, eig0)
                
                #Fourier Transform
                qG = pd.G_Qv[pd.Q_qG[0]] + pd.K_qv
                coef_G = np.diagonal(np.inner(qG, qG)) / (4 * pi)
                qGr_R = np.inner(qG, r.T).T
                phase = np.exp(1j * qGr_R)
                v_ind = np.append(v_ind,
                                  np.dot(phase, vec[:, k])[np.newaxis, :],
                                  axis=0)
                n_ind = np.append(n_ind,
                                  np.dot(phase, vec[:, k] *
                                         coef_G)[np.newaxis, :],
                                  axis=0)
        
        if name is None and self.name:
            name = (self.name + '%+d%+d%+d-eigenmodes.pckl' %
                    tuple((q_c * kd.N_c).round()))
        elif name:
            name = (name + '%+d%+d%+d-eigenmodes.pckl' %
                    tuple((q_c * kd.N_c).round()))
        else:
            name = '%+d%+d%+d-eigenmodes.pckl' % tuple((q_c * kd.N_c).round())
        
        # Returns: real space grid, frequency grid, all eigenvalues,
        # sorted eigenvalues, zero-crossing frequencies + eigenvalues,
        # induced potential + density in real space.
        if eigenvalue_only:
            pickle.dump((r * Bohr, w_w, eig_all),
                        open(name, 'wb'), pickle.HIGHEST_PROTOCOL)
            return r * Bohr, w_w, eig_all
        else:
            pickle.dump((r * Bohr, w_w, eig_all, eig, omega0, eigen0,
                         v_ind, n_ind),
                        open(name, 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            return r * Bohr, w_w, eig_all, eig, omega0, eigen0, v_ind, n_ind
    
    def get_spatial_eels(self, q_c=[0, 0, 0], direction='x',
                         w_max=None, filename='eels'):
        """Spatially resolved loss spectrum.
        
        The spatially resolved loss spectrum is calculated as the inverse
        fourier transform of ``VChiV = (eps^{-1}-I)V``::
            
            EELS(w,r) = - Im [sum_{G,G'} e^{iGr} Vchi_{GG'}(w) V_G'e^{-iG'r}]
                          \delta(w-G\dot v_e )
        
        Input parameters:
            
        direction: 'x', 'y', or 'z'
            The direction for scanning acroos the structure
            (perpendicular to the electron beam) .
        w_max: float
            maximum frequency
        filename: str
            name of output
            
        Returns: real space grid, frequency points, EELS(w,r)
        """

        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.calculate_chi0(q_c)
        e_wGG = self.get_dielectric_matrix(xc='RPA', q_c=q_c,
                                           wigner_seitz_truncation=True,
                                           symmetric=False)
        r = pd.gd.get_grid_point_coordinates()
        ix = r.shape[1] / 2
        iy = r.shape[2] / 2
        iz = r.shape[3] / 2
        if direction == 'x':
            r = r[:, :, iy, iz]
            perpdir = [1, 2]
        if direction == 'y':
            r = r[:, ix, :, iz]
            perpdir = [0, 2]
        if direction == 'z':
            r = r[:, ix, iy, :]
            perpdir = [0, 1]

        nG = e_wGG.shape[1]
        Gvec = pd.G_Qv[pd.Q_qG[0]]
        Glist = []
        # Only use G-vectors that are zero along electron beam
        # due to \delta(w-G\dot v_e )
        for iG in range(nG):
            if Gvec[iG, perpdir[0]] == 0 and Gvec[iG, perpdir[1]] == 0:
                Glist.append(iG)
        qG = Gvec[Glist] + pd.K_qv
        w_w = self.chi0.omega_w * Hartree
        if w_max:
            w_w = w_w[np.where(w_w < w_max)]
        Nw = len(w_w)
        qGr = np.inner(qG, r.T).T
        phase = np.exp(1j * qGr)
        
        V_G = (4 * pi) / np.diagonal(np.inner(qG, qG))
        phase2 = np.exp(-1j * qGr) * V_G
        E_wrr = np.zeros([Nw, r.shape[1], r.shape[1]])
        E_wr = np.zeros([Nw, r.shape[1]])
        for i in range(Nw):
            Vchi_GG = (np.linalg.inv(
                        e_wGG[i, np.array(Glist), :][:, np.array(Glist)]) -
                       np.eye(len(Glist)))
            # Fourier transform:
            E_wrr[i] = -np.imag(np.dot(np.dot(phase, Vchi_GG), phase2.T))
            E_wr[i] = np.diagonal(E_wrr[i])
        pickle.dump((r * Bohr, w_w, E_wr), open('%s.pickle' % filename, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
                    
        return r * Bohr, w_w, E_wr
