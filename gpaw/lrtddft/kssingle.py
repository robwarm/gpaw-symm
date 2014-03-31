"""Kohn-Sham single particle excitations realated objects.

"""
from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Hartree, alpha

import gpaw.mpi as mpi
from gpaw.utilities import packed_index
from gpaw.lrtddft.excitation import Excitation, ExcitationList
from gpaw.localized_functions import create_localized_functions
from gpaw.pair_density import PairDensity
from gpaw.fd_operators import Gradient
from gpaw.gaunt import gaunt as G_LLL
from gpaw.xc.tools import vxc
from gpaw.utilities.tools import coordinates

class KSSingles(ExcitationList):
    """Kohn-Sham single particle excitations

    Input parameters:

    calculator:
      the calculator object after a ground state calculation
      
    nspins:
      number of spins considered in the calculation
      Note: Valid only for unpolarised ground state calculation

    eps:
      Minimal occupation difference for a transition (default 0.001)

    istart:
      First occupied state to consider
    jend:
      Last unoccupied state to consider
    energy_range:
      The energy range [emin, emax] or emax for KS transitions to use as basis
    """

    def __init__(self,
                 calculator=None,
                 nonselfconsistent_xc=None,
                 nspins=None,
                 eps=0.001,
                 istart=0,
                 jend=None,
                 energy_range=None,
                 filehandle=None,
                 txt=None):

        if filehandle is not None:
            self.read(fh=filehandle)
            return None

        ExcitationList.__init__(self, calculator, txt=txt)
        
        if calculator is None:
            return # leave the list empty

        # deny hybrids as their empty states are wrong
        gsxc = calculator.hamiltonian.xc
        hybrid = hasattr(gsxc, 'hybrid') and gsxc.hybrid > 0.0
#        assert(not hybrid)

        if nonselfconsistent_xc is None:
            self.de_skn = None
        else:
            pass
#            self.de_skn = (vxc(calculator, nonselfconsistent_xc.name) - 
#                           vxc(calculator, calculator.hamiltonian.xc))

        error = calculator.wfs.eigensolver.error
        criterion = (calculator.input_parameters['convergence']['eigenstates']
                     * calculator.wfs.nvalence)
        if error > criterion:
            raise RuntimeError('The wfs error is larger than ' +
                               'the convergence criterion (' +
                               str(error) + ' > ' + str(criterion) + ')')

        self.select(nspins, eps, istart, jend, energy_range)

        trkm = self.get_trk()
        print >> self.txt, 'KSS TRK sum %g (%g,%g,%g)' % \
              (np.sum(trkm)/3., trkm[0], trkm[1], trkm[2])
        pol = self.get_polarizabilities(lmax=3)
        print >> self.txt, \
              'KSS polarisabilities(l=0-3) %g, %g, %g, %g' % \
              tuple(pol.tolist())

    def select(self, nspins=None, eps=0.001,
               istart=0, jend=None, energy_range=None):
        """Select KSSingles according to the given criterium."""

        paw = self.calculator
        wfs = paw.wfs
        self.kpt_u = wfs.kpt_u

        if self.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions in calculator!')

        # here, we need to take care of the spins also for
        # closed shell systems (Sz=0)
        # vspin is the virtual spin of the wave functions,
        #       i.e. the spin used in the ground state calculation
        # pspin is the physical spin of the wave functions
        #       i.e. the spin of the excited states
        self.nvspins = wfs.nspins
        self.npspins = wfs.nspins
        fijscale = 1
        if self.nvspins < 2:
            if nspins > self.nvspins:
                self.npspins = nspins
                fijscale = 0.5

        if energy_range is not None:
            try:
                emin, emax = energy_range
            except:
                emax = energy_range
                emin = 0.
            emin /= Hartree
            emax /= Hartree
            # select transitions according to transition energy
            for ispin in range(self.npspins):
                vspin = ispin
                if self.nvspins < 2:
                    vspin = 0
                kpt = self.kpt_u[vspin]
                f_n = kpt.f_n
                eps_n = kpt.eps_n
                if self.de_skn is not None:
                    eps_n += de_skn[kpt.s, kpt.k]
                for i in range(len(f_n)):
                    for j in range(i+1, len(f_n)):
                        fij = f_n[i] - f_n[j]
                        epsij = eps_n[j] - eps_n[i]
                        if fij > eps and epsij >= emin and epsij < emax:
                            # this is an accepted transition
                            ks = KSSingle(i, j, ispin, kpt, paw,
                                          fijscale = fijscale)
                            self.append(ks)
            self.istart = 0
            self.jend = -1
        else:
            # select transitions according to band index
            for ispin in range(self.npspins):
                vspin = ispin
                if self.nvspins < 2:
                    vspin = 0
                f = self.kpt_u[vspin].f_n
                if jend == None: jend = len(f)-1
                else         : jend = min(jend, len(f)-1)

                for i in range(istart, jend+1):
                    for j in range(istart, jend+1):
                        fij = f[i]-f[j]
                        if fij > eps:
                            # this is an accepted transition
                            ks = KSSingle(i, j, ispin, 
                                          self.kpt_u[vspin], paw,
                                          fijscale=fijscale)
                            self.append(ks)

            self.istart = istart
            self.jend = jend

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if fh is None:
            if filename.endswith('.gz'):
                import gzip
                f = gzip.open(filename)
            else:
                f = open(filename, 'r')
        else:
            f = fh

        f.readline()
        n = int(f.readline())
        self.npspins = 1
        for i in range(n):
            kss = KSSingle(string = f.readline())
            self.append(kss)
            self.npspins = max(self.npspins, kss.pspin + 1)
        self.update()

        if fh is None:
            f.close()

    def update(self):
        istart = self[0].i
        jend = 0
        npspins = 1
        nvspins = 1
        for kss in self:
            istart = min(kss.i, istart)
            jend = max(kss.j, jend)
            if kss.pspin == 1:
                npspins = 2
            if kss.spin == 1:
                nvspins = 2
        self.istart = istart
        self.jend = jend
        self.npspins = npspins
        self.nvspins = nvspins

        if hasattr(self, 'energies'):
            del(self.energies)

    def set_arrays(self):
        if hasattr(self, 'energies'):
            return
        energies = []
        fij = []
        me = []
        mur = []
        muv = []
        magn = []
        for k in self:
            energies.append(k.energy)
            fij.append(k.fij)
            me.append(k.me)
            mur.append(k.mur)
            if k.muv is not None:
                muv.append(k.muv)
            if k.magn is not None:
                magn.append(k.magn)
        self.energies = np.array(energies)
        self.fij = np.array(fij)
        self.me = np.array(me)
        self.mur = np.array(mur)
        if len(muv):
            self.muv = np.array(muv)
        else:
            self.muv = None
        if len(magn):
            self.magn = np.array(magn)
        else:
            self.magn = None

    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files.
        """
        if mpi.rank == mpi.MASTER:
            if fh is None:
                if filename.endswith('.gz'):
                    import gzip
                    f = gzip.open(filename,'wb')
                else:
                    f = open(filename, 'w')
            else:
                f = fh

            f.write('# KSSingles\n')
            f.write('%d\n' % len(self))
            for kss in self:
                f.write(kss.outstring())
            
            if fh is None:
                f.close()

 
class KSSingle(Excitation, PairDensity):
    """Single Kohn-Sham transition containing all it's indicees

    ::

      pspin=physical spin
      spin=virtual  spin, i.e. spin in the ground state calc.
      kpt=the Kpoint object
      fijscale=weight for the occupation difference::
      me  = sqrt(fij*epsij) * <i|r|j>
      mur = - <i|r|a>
      muv = - <i|nabla|a>/omega_ia with omega_ia>0
      magn = <i|[r x nabla]|a> / (2 m_e c)
    """
    def __init__(self, iidx=None, jidx=None, pspin=None, kpt=None,
                 paw=None, string=None, fijscale=1):
        
        if string is not None: 
            self.fromstring(string)
            return None

        # normal entry
        
        PairDensity.__init__(self, paw)
        wfs = paw.wfs
        PairDensity.initialize(self, kpt, iidx, jidx)

        self.pspin=pspin
        
        f = kpt.f_n
        self.fij = (f[iidx] - f[jidx]) * fijscale
        e = kpt.eps_n
        self.energy = e[jidx] - e[iidx]

        # calculate matrix elements -----------

        gd = wfs.gd
        self.gd = gd

        # length form ..........................

        # course grid contribution
        # <i|r|j> is the negative of the dipole moment (because of negative
        # e- charge)
        me = - gd.calculate_dipole_moment(self.get())

        # augmentation contributions
        ma = np.zeros(me.shape)
        pos_av = paw.atoms.get_positions() / Bohr
        for a, P_ni in kpt.P_ani.items():
            Ra = pos_av[a]
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            Delta_pL = wfs.setups[a].Delta_pL
            ni=len(Pi_i)
            ma0 = 0
            ma1 = np.zeros(me.shape)
            for i in range(ni):
                for j in range(ni):
                    pij = Pi_i[i]*Pj_i[j]
                    ij = packed_index(i, j, ni)
                    # L=0 term
                    ma0 += Delta_pL[ij,0]*pij
                    # L=1 terms
                    if wfs.setups[a].lmax >= 1:
                        # see spherical_harmonics.py for
                        # L=1:y L=2:z; L=3:x
                        ma1 += np.array([Delta_pL[ij,3], Delta_pL[ij,1],
                                         Delta_pL[ij,2]]) * pij
            ma += sqrt(4 * pi / 3) * ma1 + Ra * sqrt(4 * pi) * ma0
        gd.comm.sum(ma)

        self.me = sqrt(self.energy * self.fij) * ( me + ma )

        self.mur = - ( me + ma )

        # velocity form .............................

        me = np.zeros(self.mur.shape)

        # get derivatives
        dtype = self.wfj.dtype
        dwfj_cg = gd.empty((3), dtype=dtype)
        if not hasattr(gd, 'ddr'):
            gd.ddr = [Gradient(gd, c, dtype=dtype).apply for c in range(3)]
        for c in range(3):
            gd.ddr[c](self.wfj, dwfj_cg[c], kpt.phase_cd)
            me[c] = gd.integrate(self.wfi * dwfj_cg[c])

        if 0:
            me2 = np.zeros(self.mur.shape)
            for c in range(3):
                gd.ddr[c](self.wfi, dwfj_cg[c], kpt.phase_cd)
                me2[c] = gd.integrate(self.wfj * dwfj_cg[c])
            print me, -me2, me2+me

        # augmentation contributions
        ma = np.zeros(me.shape)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            nabla_iiv = paw.wfs.setups[a].nabla_iiv
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma[c] += Pi * Pj * nabla_iiv[i1, i2, c]
        gd.comm.sum(ma)
        
        self.muv = - (me + ma) / self.energy
##        print self.mur, self.muv, self.mur - self.muv

        # magnetic transition dipole ................

        magn = np.zeros(me.shape)
        r_cg, r2_g = coordinates(gd)

        wfi_g = self.wfi
        for ci in range(3):
            cj = (ci + 1) % 3
            ck = (ci + 2) % 3
            magn[ci] = gd.integrate(wfi_g * r_cg[cj] * dwfj_cg[ck] -
                                    wfi_g * r_cg[ck] * dwfj_cg[cj]  )
        # augmentation contributions
        ma = np.zeros(magn.shape)
        for a, P_ni in kpt.P_ani.items():
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            rnabla_iiv = paw.wfs.setups[a].rnabla_iiv
            for c in range(3):
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pj in enumerate(Pj_i):
                        ma[c] += Pi * Pj * rnabla_iiv[i1, i2, c]
        gd.comm.sum(ma)
        
        self.magn = -alpha / 2. * (magn + ma)

    def __add__(self, other):
        """Add two KSSingles"""
        result = self.copy()
        result.me = self.me + other.me
        result.mur = self.mur + other.mur
        result.muv = self.muv + other.muv
        return result

    def __sub__(self, other):
        """Subtract two KSSingles"""
        result = self.copy()
        result.me = self.me - other.me
        result.mur = self.mur - other.mur
        result.muv = self.muv - other.muv
        return result

    def __mul__(self, x):
        """Multiply a KSSingle with a number"""
        if type(x) == type(0.) or type(x) == type(0):
            result = self.copy()
            result.me = self.me * x
            result.mur = self.mur * x
            result.muv = self.muv * x
            return result
        else:
            return RuntimeError('not a number')
        
    def __div__(self, x):
        return self.__mul__(1. / x)

    def copy(self):
        return KSSingle(string=self.outstring())

    def fromstring(self,string):
        l = string.split()
        self.i = int(l.pop(0))
        self.j = int(l.pop(0))
        self.pspin = int(l.pop(0))
        self.spin = int(l.pop(0))
        self.energy = float(l.pop(0))
        self.fij = float(l.pop(0))
        if len(l) == 3: # old writing style
            self.me = np.array([float(l.pop(0)) for i in range(3)])
            self.mur = - self.me / sqrt(self.energy * self.fij)
            self.muv = None
            self.magn = None
        else:
            self.mur = np.array([float(l.pop(0)) for i in range(3)])
            self.me = - self.mur * sqrt(self.energy * self.fij)
            self.muv = np.array([float(l.pop(0)) for i in range(3)])
            if len(l): 
                self.magn = np.array([float(l.pop(0)) for i in range(3)])
            else:
                self.magn = None
        return None

    def outstring(self):
        str = '%d %d   %d %d   %g %g' % \
               (self.i,self.j, self.pspin,self.spin, self.energy, self.fij)
        str += '  '
        for m in self.mur: str += '%12.4e' % m
        str += '  '
        for m in self.muv: str += '%12.4e' % m
        str += '  '
        for m in self.magn: str += '%12.4e' % m
        str += '\n'
        return str
        
    def __str__(self):
        str = "# <KSSingle> %d->%d %d(%d) eji=%g[eV]" % \
              (self.i, self.j, self.pspin, self.spin,
               self.energy * Hartree)
        str += " (%g,%g,%g)" % (self.me[0], self.me[1], self.me[2])
        return str
    
    #####################
    ## User interface: ##
    #####################

    def get_weight(self):
        return self.fij

