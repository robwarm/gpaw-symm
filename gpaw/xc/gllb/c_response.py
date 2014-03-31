from gpaw.xc.gllb.contribution import Contribution
#from gpaw.xc_functional import XCRadialGrid, XCFunctional, XC3DGrid
#from gpaw.xc_correction import A_Liy, weights
from gpaw.sphere.lebedev import weight_n
from gpaw.utilities import pack
from gpaw.xc.gllb import safe_sqr
from math import sqrt, pi
from gpaw.mpi import world
from ase.units import Hartree
import numpy as np

class C_Response(Contribution):
    def __init__(self, nlfunc, weight, coefficients):
        Contribution.__init__(self, nlfunc, weight)
        self.coefficients = coefficients
        self.vt_sg = None
        self.vt_sG = None
        self.nt_sG = None
        self.D_asp = None
        self.Dresp_asp = None

    def get_name(self):
        return "RESPONSE"

    def get_desc(self):
        return ""
        
    # Initialize Response functional
    def initialize_1d(self):
        self.ae = self.nlfunc.ae

    # Calcualte the GLLB potential and energy 1d
    def add_xc_potential_and_energy_1d(self, v_g):
        w_i = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g += self.weight * np.dot(w_i, u2_j) / (np.dot(self.ae.f_j, u2_j) +1e-10)
        return 0.0 # Response part does not contribute to energy

    def initialize(self):
        self.gd = self.nlfunc.gd
        self.finegd = self.nlfunc.finegd
        self.wfs = self.nlfunc.wfs
        self.kpt_u = self.wfs.kpt_u
        self.setups = self.wfs.setups
        self.density = self.nlfunc.density
        self.symmetry = self.wfs.symmetry
        self.nspins = self.nlfunc.nspins
        self.occupations = self.nlfunc.occupations
        self.nvalence = self.nlfunc.nvalence
        self.kpt_comm = self.wfs.kpt_comm
        self.band_comm = self.wfs.band_comm
        self.grid_comm = self.gd.comm
        if self.vt_sg is None or self.density is None \
           or self.density.nt_sg is None or \
            np.any(self.vt_sg.shape != self.density.nt_sg.shape):
             self.vt_sg = self.finegd.empty(self.nlfunc.nspins)
             self.vt_sG = self.gd.empty(self.nlfunc.nspins)
             self.nt_sG = self.gd.empty(self.nlfunc.nspins)
        if self.Dresp_asp is None:
            self.Dresp_asp = {}
            self.D_asp = {}
            if self.density.D_asp is not None:
                for a in self.density.D_asp:
                    self.Dresp_asp[a] = np.zeros_like(self.density.D_asp[a])
                    self.D_asp[a] = np.zeros_like(self.density.D_asp[a])

        # The response discontinuity is stored here
        self.Dxc_vt_sG = None
        self.Dxc_Dresp_asp = {}
        self.Dxc_D_asp = {}

    def update_potentials(self, nt_sg):
        nspins = len(nt_sg)
        w_kn = self.coefficients.get_coefficients_by_kpt(self.kpt_u, nspins=nspins)
        f_kn = [ kpt.f_n for kpt in self.kpt_u ]

        if w_kn is not None:
            self.vt_sG[:] = 0.0
            self.nt_sG[:] = 0.0

            # XXX This is terribly inefficients with LCAO basis set
            # XXX Form a "response" density-matrix first before expanding the numerator of the response potential.
            for kpt, w_n in zip(self.kpt_u, w_kn):
                self.wfs.add_to_density_from_k_point_with_occupation(self.vt_sG, kpt, w_n)
                self.wfs.add_to_density_from_k_point(self.nt_sG, kpt)

            self.band_comm.sum(self.nt_sG)
            self.kpt_comm.sum(self.nt_sG)
            self.band_comm.sum(self.vt_sG)
            self.kpt_comm.sum(self.vt_sG)

            if self.wfs.symmetry:
                for nt_G, vt_G in zip(self.nt_sG, self.vt_sG):
                    self.symmetry.symmetrize(nt_G, self.gd)
                    self.symmetry.symmetrize(vt_G, self.gd)

            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.Dresp_asp, w_kn)
            self.wfs.calculate_atomic_density_matrices_with_occupation(
                self.D_asp, f_kn)

            self.vt_sG /= self.nt_sG +1e-10
        self.density.interpolate(self.vt_sG, self.vt_sg)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        self.update_potentials([n_g]) 
        v_g[:] += self.weight * self.vt_sg[0]
        return 0.0

    def calculate_spinpolarized(self, e_g, n_sg, v_sg):
        self.update_potentials(n_sg)
        v_sg += self.weight * self.vt_sg
        return 0.0

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a, addcoredensity=True):
        # Get the XC-correction instance
        c = setup.xc_correction

        ncresp_g = setup.extra_xc_data['core_response'] / self.nspins
        if not addcoredensity:
            ncresp_g[:] = 0.0
        
        for D_p, dEdD_p, Dresp_p in zip(D_sp, H_sp, self.Dresp_asp.get(a)):
            D_Lq = np.dot(c.B_pqL.T, D_p)
            n_Lg = np.dot(D_Lq, c.n_qg) # Construct density
            if addcoredensity:
                n_Lg[0] += c.nc_g * sqrt(4 * pi) / self.nspins
            nt_Lg = np.dot(D_Lq, c.nt_qg) # Construct smooth density (_without smooth core density_)

            Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p)
            nresp_Lg = np.dot(Dresp_Lq, c.n_qg) # Construct 'response density'
            nrespt_Lg = np.dot(Dresp_Lq, c.nt_qg) # Construct smooth 'response density' (w/o smooth core)

            for w, Y_L in zip(weight_n, c.Y_nL):
                nt_g = np.dot(Y_L, nt_Lg)
                nrespt_g = np.dot(Y_L, nrespt_Lg)
                x_g = nrespt_g / (nt_g + 1e-10)
                dEdD_p -= self.weight * w * np.dot(np.dot(c.B_pqL, Y_L),
                                                   np.dot(c.nt_qg, x_g * c.rgd.dv_g))

                n_g = np.dot(Y_L, n_Lg)
                nresp_g = np.dot(Y_L, nresp_Lg)
                x_g = (nresp_g+ncresp_g) / (n_g + 1e-10)
           
                dEdD_p += self.weight * w * np.dot(np.dot(c.B_pqL, Y_L),
                                                   np.dot(c.n_qg, x_g * c.rgd.dv_g))
            
        return 0.0

    def integrate_sphere(self, a, Dresp_sp, D_sp, Dwf_p):
        c = self.nlfunc.setups[a].xc_correction
        Dresp_p, D_p = Dresp_sp[0], D_sp[0]
        D_Lq = np.dot(c.B_pqL.T, D_p)
        n_Lg = np.dot(D_Lq, c.n_qg) # Construct density
        n_Lg[0] += c.nc_g * sqrt(4 * pi)
        nt_Lg = np.dot(D_Lq, c.nt_qg) # Construct smooth density (without smooth core)
        Dresp_Lq = np.dot(c.B_pqL.T, Dresp_p) # Construct response
        nresp_Lg = np.dot(Dresp_Lq, c.n_qg) # Construct 'response density'
        nrespt_Lg = np.dot(Dresp_Lq, c.nt_qg) # Construct smooth 'response density' (w/o smooth core)
        Dwf_Lq = np.dot(c.B_pqL.T, Dwf_p) # Construct lumo wf
        nwf_Lg = np.dot(Dwf_Lq, c.n_qg)
        nwft_Lg = np.dot(Dwf_Lq, c.nt_qg)
        E = 0.0
        for w, Y_L in zip(weight_n, c.Y_nL):
            v = np.dot(Y_L, nwft_Lg) * np.dot(Y_L, nrespt_Lg) / (np.dot(Y_L, nt_Lg) + 1e-10)
            E -= self.weight * w * np.dot(v, c.rgd.dv_g)
            v = np.dot(Y_L, nwf_Lg) * np.dot(Y_L, nresp_Lg) / (np.dot(Y_L, n_Lg) + 1e-10)
            E += self.weight * w * np.dot(v, c.rgd.dv_g)
        return E

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        v_g = np.zeros(self.ae.N)
        n_g = np.zeros(self.ae.N)
        for w_n, f_n, u_n in zip(w_ln, self.ae.f_ln, self.ae.s_ln): # For each angular momentum
            u2_n = safe_sqr(u_n)
            v_g += np.dot(w_n, u2_n)
            n_g += np.dot(f_n, u2_n)
                           
        vt_g += self.weight * v_g / (n_g + 1e-10)
        return 0.0 # Response part does not contribute to energy

    def calculate_delta_xc(self, homolumo = None):
        if homolumo == None:
            # Calculate band gap
            print "Warning: Calculating KS-gap directly from the k-points, can be inaccurate."
            #homolumo = self.occupations.get_homo_lumo(self.wfs)

        #homo, lumo = homolumo
        #Ksgap = lumo-homo
        #print "Using KS-gap of ", Ksgap
        
        for a in self.density.D_asp:
            ni = self.setups[a].ni
            self.Dxc_Dresp_asp[a] = np.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            self.Dxc_D_asp[a] = np.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))

        # Calculate new response potential with LUMO reference 
        w_kn = self.coefficients.get_coefficients_by_kpt(self.kpt_u, lumo_perturbation=True, 
                                                         homolumo=homolumo,
                                                         nspins=self.nspins)
        f_kn = [ kpt.f_n for kpt in self.kpt_u ]

        vt_sG = self.gd.zeros(self.nlfunc.nspins)
        nt_sG = self.gd.zeros(self.nlfunc.nspins)
        for kpt, w_n in zip(self.kpt_u, w_kn):
            self.wfs.add_to_density_from_k_point_with_occupation(vt_sG, kpt, w_n)
            self.wfs.add_to_density_from_k_point(nt_sG, kpt)
            
        self.band_comm.sum(nt_sG)
        self.kpt_comm.sum(nt_sG)
        self.band_comm.sum(vt_sG)
        self.kpt_comm.sum(vt_sG)
            
        if self.wfs.symmetry:
            for nt_G, vt_G in zip(nt_sG, vt_sG):
                self.symmetry.symmetrize(nt_G, self.gd)
                self.symmetry.symmetrize(vt_G, self.gd)

        vt_sG /= nt_sG + 1e-10
        self.Dxc_vt_sG = vt_sG.copy()

        self.wfs.calculate_atomic_density_matrices_with_occupation(
            self.Dxc_Dresp_asp, w_kn)
        self.wfs.calculate_atomic_density_matrices_with_occupation(
            self.Dxc_D_asp, f_kn)

    def calculate_delta_xc_perturbation_spin(self, s=0):
        homo, lumo = self.occupations.get_homo_lumo_by_spin(self.wfs, s)
        Ksgap = lumo - homo
        
        # Calculate average of lumo reference response potential
        method1_dxc = np.average(self.Dxc_vt_sG[s])
        nt_G = self.gd.empty()

        epsilon = 1e-2
        gaps = [ 1000.0 ]
        for u, kpt in enumerate(self.kpt_u):
            if kpt.s == s:
                lumo_n = np.flatnonzero(kpt.f_n/kpt.weight <= epsilon)[0]
                nt_G[:] = 0.0
                self.wfs.add_orbital_density(nt_G, kpt, lumo_n)
                E = 0.0
                for a in self.density.D_asp:
                    D_sp = self.Dxc_D_asp[a]
                    Dresp_sp = self.Dxc_Dresp_asp[a]
                    P_ni = kpt.P_ani[a]
                    Dwf_p = pack(np.outer(P_ni[lumo_n].T.conj(), P_ni[lumo_n]).real)
                    E += self.integrate_sphere(a, Dresp_sp, D_sp, Dwf_p)
                E = self.grid_comm.sum(E)
                E += self.gd.integrate(nt_G*self.Dxc_vt_sG[s])
                E += kpt.eps_n[lumo_n]
                gaps.append(E-lumo)

        method2_dxc = self.kpt_comm.min(min(gaps))
        Ksgap *= Hartree
        method1_dxc *= Hartree
        method2_dxc *= Hartree
        if world.rank is not 0:
            return (Ksgap, method2_dxc)
        
        print
        print "\Delta XC calulation"
        print "-----------------------------------------------"
        print "| Method      |  KS-Gap | \Delta XC |  QP-Gap |"
        print "-----------------------------------------------"
        print "| Averaging   | %7.2f | %9.2f | %7.2f |" % (Ksgap, method1_dxc, Ksgap+method1_dxc)
        print "| Lumo pert.  | %7.2f | %9.2f | %7.2f |" % (Ksgap, method2_dxc, Ksgap+method2_dxc)
        print "-----------------------------------------------"
        print
        return (Ksgap, method2_dxc)


    def calculate_delta_xc_perturbation(self):
        gaps = []
        for s in range(0, self.nspins):
            gaps.append(self.calculate_delta_xc_perturbation_spin(s))
        if self.nspins == 1:
            return gaps[0]
        else:
            return gaps

    def initialize_from_atomic_orbitals(self, basis_functions):
        # Initiailze 'response-density' and density-matrices
        self.Dresp_asp = {}
        self.D_asp = {}
        
        for a in self.density.D_asp.keys():
            ni = self.setups[a].ni
            self.Dresp_asp[a] = np.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            self.D_asp[a] = np.zeros((self.nlfunc.nspins, ni * (ni + 1) // 2))
            
        f_sM = np.empty((self.nspins, basis_functions.Mmax))
        self.D_asp = {}
        f_asi = {}
        w_asi = {}              

        for a in basis_functions.atom_indices:
            w_j = self.setups[a].extra_xc_data['w_j']
            # Basis function coefficients based of response weights
            w_si = self.setups[a].calculate_initial_occupation_numbers(
                    0, False, charge=0, nspins=self.nspins, f_j=w_j)
            # Basis function coefficients based on density
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                    0, False, charge=0, nspins=self.nspins)            

            if a in basis_functions.my_atom_indices:
                self.Dresp_asp[a] = self.setups[a].initialize_density_matrix(w_si)
                self.D_asp[a] = self.setups[a].initialize_density_matrix(f_si)
                
            f_asi[a] = f_si
            w_asi[a] = w_si

        self.nt_sG.fill(0.0)
        basis_functions.add_to_density(self.nt_sG, f_asi)
        self.vt_sG.fill(0.0)
        basis_functions.add_to_density(self.vt_sG, w_asi)
        # Update vt_sG to correspond atomic response potential. This will be
        # used until occupations and eigenvalues are available.
        self.vt_sG /= self.nt_sG + 1e-10

    def add_extra_setup_data(self, dict):
        ae = self.ae
        njcore = ae.njcore
        w_ln = self.coefficients.get_coefficients_1d(smooth=True)
        w_j = []
        for w_n in w_ln:
            for w in w_n:
                w_j.append(w)
        dict['w_j'] = w_j

        w_j = self.coefficients.get_coefficients_1d()
        x_g = np.dot(w_j[:njcore], safe_sqr(ae.u_j[:njcore]))
        x_g[1:] /= ae.r[1:]**2 * 4*np.pi
        x_g[0] = x_g[1]
        dict['core_response'] = x_g

        # For debugging purposes
        w_j = self.coefficients.get_coefficients_1d()
        u2_j = safe_sqr(self.ae.u_j)
        v_g = self.weight * np.dot(w_j, u2_j) / (np.dot(self.ae.f_j, u2_j) +1e-10)
        v_g[0] = v_g[1]
        dict['all_electron_response'] = v_g

        # Calculate Hardness of spherical atom, for debugging purposes
        l = [ np.where(f<1e-3, e, 1000) for f,e in zip(self.ae.f_j, self.ae.e_j)]
        h = [ np.where(f>1e-3, e, -1000) for f,e in zip(self.ae.f_j, self.ae.e_j)]
        lumo_e = min(l)
        homo_e = max(h)
        if lumo_e < 999: # If there is unoccpied orbital
            w_j = self.coefficients.get_coefficients_1d(lumo_perturbation = True)
            v_g = self.weight * np.dot(w_j, u2_j) / (np.dot(self.ae.f_j, u2_j) +1e-10)
            e2 = [ e+np.dot(u2*v_g, self.ae.dr) for u2,e in zip(u2_j, self.ae.e_j) ]
            lumo_2 = min([ np.where(f<1e-3, e, 1000) for f,e in zip(self.ae.f_j, e2)])
            print "New lumo eigenvalue:", lumo_2 * 27.2107
            self.hardness = lumo_2 - homo_e
            print "Hardness predicted: %10.3f eV" % (self.hardness * 27.2107)
            
    def write(self, w, natoms):
        """Writes response specific data to disc.
        
        During the writing process, the DeltaXC is calculated (if not yet calculated)
        """
        
        if self.Dxc_vt_sG == None:
            self.calculate_delta_xc()
        
        wfs = self.wfs
        world = wfs.world
        domain_comm = wfs.gd.comm
        kpt_comm = wfs.kpt_comm
        band_comm = wfs.band_comm
        
        master = (world.rank == 0)

        nadm = 0
        for setup in wfs.setups:
            ni = setup.ni
            nadm += ni * (ni + 1) // 2

        # Not yet tested for parallerization
        #assert world.size == 1

        # Write the pseudodensity on the coarse grid:
        if master:
            w.add('GLLBPseudoResponsePotential',
                  ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)
        if kpt_comm.rank == 0:
            for s in range(wfs.nspins):
                vt_sG = wfs.gd.collect(self.vt_sG[s])
                if master:
                    w.fill(vt_sG)

        if master:
            w.add('GLLBDxcPseudoResponsePotential',
                  ('nspins', 'ngptsx', 'ngptsy', 'ngptsz'), dtype=float)

        if kpt_comm.rank == 0:
            for s in range(wfs.nspins):
                vt_sG = wfs.gd.collect(self.Dxc_vt_sG[s])
                if master:
                    w.fill(vt_sG)

        print "Integration over vt_sG", domain_comm.sum(np.sum(self.vt_sG.ravel()))
        print "Integration over Dxc_vt_sG", domain_comm.sum(np.sum(self.Dxc_vt_sG.ravel()))
                
        if master:
            all_D_sp = np.empty((wfs.nspins, nadm))
            all_Dresp_sp = np.empty((wfs.nspins, nadm))
            all_Dxc_D_sp = np.empty((wfs.nspins, nadm))
            all_Dxc_Dresp_sp = np.empty((wfs.nspins, nadm))
            p1 = 0
            for a in range(natoms):
                ni = wfs.setups[a].ni
                nii = ni * (ni + 1) // 2
                if a in self.D_asp:
                    D_sp = self.D_asp[a]
                    Dresp_sp = self.Dresp_asp[a]
                    Dxc_D_sp = self.Dxc_D_asp[a]
                    Dxc_Dresp_sp = self.Dxc_Dresp_asp[a]
                else:
                    D_sp = np.empty((wfs.nspins, nii))
                    domain_comm.receive(D_sp, wfs.rank_a[a], 27)
                    Dresp_sp = np.empty((wfs.nspins, nii))
                    domain_comm.receive(Dresp_sp, wfs.rank_a[a], 271)
                    Dxc_D_sp = np.empty((wfs.nspins, nii))
                    domain_comm.receive(Dxc_D_sp, wfs.rank_a[a], 28)
                    Dxc_Dresp_sp = np.empty((wfs.nspins, nii))
                    domain_comm.receive(Dxc_Dresp_sp, wfs.rank_a[a], 272)
                    
                p2 = p1 + nii
                all_D_sp[:, p1:p2] = D_sp
                all_Dresp_sp[:, p1:p2] = Dresp_sp
                all_Dxc_D_sp[:, p1:p2] = Dxc_D_sp
                all_Dxc_Dresp_sp[:, p1:p2] = Dxc_Dresp_sp
                
                p1 = p2
            assert p2 == nadm
            w.add('GLLBAtomicDensityMatrices', ('nspins', 'nadm'), all_D_sp)
            w.add('GLLBAtomicResponseMatrices', ('nspins', 'nadm'), all_Dresp_sp)
            w.add('GLLBDxcAtomicDensityMatrices', ('nspins', 'nadm'), all_Dxc_D_sp)
            w.add('GLLBDxcAtomicResponseMatrices', ('nspins', 'nadm'), all_Dxc_Dresp_sp)
        elif kpt_comm.rank == 0 and band_comm.rank == 0:
            for a in range(natoms):
                if a in self.density.D_asp:
                    domain_comm.send(self.D_asp[a], 0, 27)
                    domain_comm.send(self.Dresp_asp[a], 0, 271)
                    domain_comm.send(self.Dxc_D_asp[a], 0, 28)
                    domain_comm.send(self.Dxc_Dresp_asp[a], 0, 272)

    def read(self, r):
        wfs = self.wfs
        world = wfs.world
        domain_comm = wfs.gd.comm
        kpt_comm = wfs.kpt_comm
        band_comm = wfs.band_comm
        
        self.vt_sG = wfs.gd.empty(wfs.nspins)
        self.Dxc_vt_sG = wfs.gd.empty(wfs.nspins)
        print "Reading vt_sG"
        for s in range(wfs.nspins):
            self.gd.distribute(r.get('GLLBPseudoResponsePotential', s),
                              self.vt_sG[s])
        print "Reading Dxc_vt_sG"
        for s in range(wfs.nspins):
            self.gd.distribute(r.get('GLLBDxcPseudoResponsePotential', s),
                              self.Dxc_vt_sG[s])
            
        print "Integration over vt_sG", domain_comm.sum(np.sum(self.vt_sG.ravel()))
        print "Integration over Dxc_vt_sG", domain_comm.sum(np.sum(self.Dxc_vt_sG.ravel()))
        
        # Read atomic density matrices and non-local part of hamiltonian:
        D_sp = r.get('GLLBAtomicDensityMatrices')
        Dresp_sp = r.get('GLLBAtomicResponseMatrices')
        Dxc_D_sp = r.get('GLLBDxcAtomicDensityMatrices')
        Dxc_Dresp_sp = r.get('GLLBDxcAtomicResponseMatrices')
        
        self.D_asp = {}
        self.Dresp_asp = {}
        self.Dxc_D_asp = {}
        self.Dxc_Dresp_asp = {}

        p1 = 0
        for a, setup in enumerate(wfs.setups):
            ni = setup.ni
            p2 = p1 + ni * (ni + 1) // 2
            # NOTE: Distrbibutes the matrices to more processors than necessary
            self.D_asp[a] = D_sp[:, p1:p2].copy()
            self.Dresp_asp[a] = Dresp_sp[:, p1:p2].copy()
            self.Dxc_D_asp[a] = Dxc_D_sp[:, p1:p2].copy()
            self.Dxc_Dresp_asp[a] = Dxc_Dresp_sp[:, p1:p2].copy()
            print "Proc", world.rank, " reading atom ", a
            p1 = p2

if __name__ == "__main__":
    from gpaw.xc_functional import XCFunctional
    xc = XCFunctional('LDA')
    dx = 1e-3
    Ntot = 100000
    x = np.array(range(1,Ntot+1))*dx
    kf = (2*x)**(1./2)
    n_g = kf**3 / (3*pi**2)
    v_g = np.zeros(Ntot)
    e_g = np.zeros(Ntot)
    xc.calculate_spinpaired(e_g, n_g, v_g)
    vresp = v_g - 2*e_g / n_g
    
    f = open('response.dat','w')
    for xx, v in zip(x, vresp):
        print >>f, xx, v
    f.close()


