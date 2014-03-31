"""Used to generate polarization functions for atomic basis sets."""

import sys
import math
import traceback

import numpy as np
from ase import Atom, Atoms
from ase.data import molecules as g2

from gpaw import Calculator
from gpaw.kpoint import KPoint
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
from gpaw.localized_functions import create_localized_functions
from gpaw.atom.all_electron import AllElectron
from gpaw.atom.configurations import configurations
from gpaw.testing.amoeba import Amoeba
from gpaw.utilities import devnull


class LinearCombination:
    """Represents a linear combination of 1D functions."""
    def __init__(self, coefs, functions):
        self.coefs = coefs
        self.functions = functions

    def __call__(self, r):
        """Evaluate function values at r, which is a numpy array."""
        return sum([coef * function(r) for coef, function
                    in zip(self.coefs, self.functions)])

    def renormalize(self, norm):
        """Divide coefficients by norm."""
        self.coefs = [coef/norm for coef in self.coefs]

def gramschmidt(gd, psit_k):
    """Orthonormalize functions on grid using the Gram-Schmidt method.

    Modifies the elements of psit_k such that each scalar product
    < psit_k[i] | psit_k[j] > = delta[ij], where psit_k are on the grid gd"""
    for k in range(len(psit_k)):
        psi = psit_k[k]
        for l in range(k):
            phi = psit_k[l]
            psit_k[k] = psi - gd.integrate(psi*phi) * phi
        psi = psit_k[k]
        psit_k[k] = psi / gd.integrate(psi*psi)**.5

def rotation_test():
    molecule = 'NH3'
    a = 7.
    rcut = 5.
    l = 1

    from gpaw.output import plot

    rotationvector = np.array([1.0, 1.0, 1.0])
    angle_increment = 0.3
    
    system = g2.molecule(molecule)
    system.set_cell([a, a, a])
    system.center()
    calc = Calculator(h=0.27, txt=None)
    system.set_calculator(calc)

    pog = PolarizationOrbitalGenerator(rcut)

    r = np.linspace(0., rcut, 300)

    maxvalues = []
    import pylab
    for i in range(0, int(6.28/angle_increment)):
        ascii = plot(system.positions,
                     system.get_atomic_numbers(),
                     system.get_cell().diagonal())

        print ascii
        
        print 'angle=%.03f' % (angle_increment * i)
        energy = system.get_potential_energy()
        center = (system.positions / system.get_cell().diagonal())[0]
        orbital = pog.generate(l, calc.wfs.gd, calc.kpt_u[0].psit_nG, center)
        y = orbital(r)
        pylab.plot(r, y, label='%.02f' % (i * angle_increment))
        maxvalues.append(max(y))
        print 'Quality by orbital', #pretty(pog.optimizer.lastterms)
        system.rotate(rotationvector, angle_increment)
        system.center()

    print max(maxvalues) - min(maxvalues)

    pylab.legend()
    pylab.show()


def make_dummy_reference(l, function=None, rcut=6., a=12., n=60,
                         dtype=float):
    """Make a mock reference wave function using a made-up radial function
    as reference"""
    #print 'Dummy reference: l=%d, rcut=%.02f, alpha=%.02f' % (l, rcut, alpha)
    r = np.arange(0., rcut, .01)

    if function is None:
        function = QuasiGaussian(4., rcut)

    norm = get_norm(r, function(r), l)
    function.renormalize(norm)
    #g = QuasiGaussian(alpha, rcut)
    
    mcount = 2*l + 1
    fcount = 1
    gd = GridDescriptor((n, n, n), (a, a, a), (False, False, False))
    spline = Spline(l, r[-1], function(r), points=50)
    center = (.5, .5, .5)
    lf = create_localized_functions([spline], gd, center, dtype=dtype)
    psit_k = gd.zeros(mcount, dtype=dtype)
    coef_xi = np.identity(mcount * fcount, dtype=dtype)
    lf.add(psit_k, coef_xi)
    return gd, psit_k, center, function

def make_dummy_kpt_reference(l, function, k_c,
                             rcut=6., a=10., n=60, dtype=complex):
    r = np.linspace(0., rcut, 300)
    mcount = 2*l + 1
    fcount = 1
    kcount = 1
    gd = GridDescriptor((n, n, n), (a, a, a), (True, True, True))
    kpt = KPoint([], gd, 1., 0, 0, 0, k_c, dtype)
    spline = Spline(l, r[-1], function(r))
    center = (.5, .5, .5)
    lf = create_localized_functions([spline], gd, center, dtype=dtype)
    lf.set_phase_factors([kpt.k_c])
    psit_nG = gd.zeros(mcount, dtype=dtype)
    coef_xi = np.identity(mcount * fcount, dtype=dtype)
    lf.add(psit_nG, coef_xi, k=0)
    kpt.psit_nG = psit_nG
    print 'Number of boxes', len(lf.box_b)
    print 'Phase kb factors shape', lf.phase_kb.shape
    return gd, kpt, center

class CoefficientOptimizer:
    """Class for optimizing Gaussian/reference overlap.

    Given matrices of scalar products s and S as returned by overlaps(),
    finds the optimal set of coefficients resulting in the largest overlap.

    ccount is the number of coefficients.
    if fix is True, the first coefficient will be set to 1, and only the
    remaining coefficients will be subject to optimization.
    """
    def __init__(self, s_kmii, S_kmii, ccount, fix=False):
        self.s_kmii = s_kmii
        self.S_kmii = S_kmii
        self.fix = fix
        function = self.evaluate
        self.lastterms = None
        if fix:
            function = self.evaluate_fixed
            ccount -= 1
        ones = np.ones((ccount, ccount))
        diag = np.identity(ccount)
        simplex = np.concatenate((np.ones((ccount,1)),
                                   ones + .5 * diag), axis=1)
        simplex = np.transpose(simplex)
        self.amoeba = Amoeba(function, simplex, tolerance=1e-10)
        
    def find_coefficients(self):
        self.amoeba.optimize()
        coefficients = self.amoeba.simplex[0]
        if self.fix:
            coefficients = [1.] + list(coefficients)
        return coefficients

    def evaluate_fixed(self, coef):
        return self.evaluate([1.] + list(coef))

    def evaluate(self, coef):
        coef = np.array(coef) # complex coefficients?
        terms_km = np.zeros(self.S_kmii.shape[0:2])
        for i, (s_mii, S_mii) in enumerate(zip(self.s_kmii, self.S_kmii)):
            for j, (s_ii, S_ii) in enumerate(zip(s_mii, S_mii)):
                numerator = np.vdot(coef, np.dot(S_ii, coef))
                denominator = np.vdot(coef, np.dot(s_ii, coef))
                terms_km[i, j] = numerator / denominator

        #print terms_km
        
        self.lastterms = terms_km
        quality = terms_km.sum()
        badness = - quality
        return badness

def norm_squared(r, f, l):
    dr = r[1]
    frl = f * r**l
    assert abs(r[1] - (r[-1] - r[-2])) < 1e-10 # error if not equidistant
    return sum(frl * frl * r * r * dr)

def get_norm(r, f, l):
    return norm_squared(r, f, l) ** .5

class PolarizationOrbitalGenerator:
    """Convenience class which generates polarization functions."""
    def __init__(self, rcut, gaussians=None):
        self.rcut = rcut
        if gaussians is None:
            gaussians = 4
        if isinstance(gaussians, int):
            self.r_alphas = np.linspace(1., .6 * rcut, gaussians + 1)[1:]
        else: # assume it is a list of actual characteristic lengths
            self.r_alphas = gaussians
        self.alphas = 1. / self.r_alphas ** 2
        self.s = None
        self.S = None
        self.optimizer = None

    def generate(self, l, gd, kpt_u, spos_ac, dtype=None):
        """Generate polarization orbital."""
        rcut = self.rcut
        phi_i = [QuasiGaussian(alpha, rcut) for alpha in self.alphas]
        r = np.arange(0, rcut, .01)
        dr = r[1] # equidistant
        integration_multiplier = r ** (2 * (l + 1))
        for phi in phi_i:
            y = phi(r)
            norm = (dr * sum(y * y * integration_multiplier)) ** .5
            phi.renormalize(norm)
        splines = [Spline(l, r[-1], phi(r)) for phi in phi_i]

        if dtype is None:
            if np.any([kpt.dtype == complex for kpt in kpt_u]):
                dtype = complex
            else:
                dtype = float

        self.s, self.S = overlaps(l, gd, splines, kpt_u, spos_ac)

        self.optimizer = CoefficientOptimizer(self.s, self.S, len(phi_i))
        coefs = self.optimizer.find_coefficients()
        self.quality = - self.optimizer.amoeba.y[0]
        self.qualities = self.optimizer.lastterms
        orbital = LinearCombination(coefs, phi_i)
        orbital.renormalize(get_norm(r, orbital(r), l))
        return orbital

def overlaps(l, gd, splines, kpt_u, spos_ac=((.5, .5, .5),),
             txt=devnull):
    """Get scalar products of basis functions and references.

    Returns the quadruple-indexed matrices s and S, where::

        s    =  < phi    | phi    > ,
         kmij        kmi      kmj

               -----
                \     /        |  ~   \   /  ~   |        \ 
        S     =  )   (  phi    | psi   ) (  psi  | phi     )
         kmij   /     \    mi  |    n /   \    n |     mj /
               -----
                 n

    The functions phi are taken from the given splines, whereas psit
    must be on the grid represented by the GridDescriptor gd.
    Integrals are evaluated at the relative location given by center.
    """
    if txt == '-':
        txt = sys.stdout

    # XXX
    spos_c = spos_ac[0]
    assert len(spos_ac) == 1, str(spos_c)

    mcount = 2 * l + 1
    fcount = len(splines)
    kcount = len(kpt_u)
    bcount = kpt_u[0].psit_nG.shape[0]

    dtype = kpt_u[0].dtype
    print >> txt, 'Creating localized functions'
    lf = create_localized_functions(splines, gd, spos_c, dtype=dtype)

    k_kc = [kpt.k_c for kpt in kpt_u]
    if dtype == complex:
        lf.set_phase_factors(k_kc)

    # make sanity checks
    for kpt in kpt_u:
        assert kpt.psit_nG.shape[0] == bcount # same band count for all kpts
    assert [kpt.dtype for kpt in kpt_u].count(dtype) == kcount # same dtype
    lvalues = [spline.get_angular_momentum_number() for spline in splines]
    assert lvalues.count(l) == len(lvalues) # all l must be equal

    # First we have to calculate the scalar products between
    # pairs of basis functions < phi_kmi | phi_kmj >.
    s_kmii = np.zeros((kcount, mcount, fcount, fcount), dtype=dtype)
    coef_xi = np.identity(mcount * fcount, dtype=dtype)
    #phi_miG = gd.zeros(mcount * fcount, dtype=dtype)
    print >> txt, 'Calculating phi-phi products'
    for kpt in kpt_u:
        gramschmidt(gd, kpt.psit_nG)
        normsqr = gd.integrate(np.conjugate(kpt.psit_nG) * kpt.psit_nG)
        for n in range(bcount):
            kpt.psit_nG[n] /= normsqr[n] ** .5
        phi_nG = gd.zeros(mcount * fcount, dtype=dtype)
        #for lf in lf_a:
        #    lf.add(phi_nG, coef_xi, k=kpt.k)
        lf.add(phi_nG, coef_xi, k=kpt.k)
        phi_overlaps_ii = np.zeros((fcount * mcount,
                                     fcount * mcount), dtype=dtype)
        # XXX products for different m unneeded.  Bottleneck for large fcount
        lf.integrate(phi_nG, phi_overlaps_ii, k=kpt.k)
        #for lf in lf_a:
        #    # every integration will add to the result array
        #    lf.integrate(phi_nG, phi_overlaps_ii, k=kpt.k)
        phi_overlaps_ii.shape = (fcount, mcount, fcount, mcount)
        for m in range(mcount):
            for i in range(fcount):
                for j in range(fcount):
                    s_kmii[kpt.u, m, i, j] = phi_overlaps_ii[i, m, j, m]

    # Now calculate scalar products between basis functions and
    # reference functions < phi_kmi | psi_kn >.
    overlaps_knmi = np.zeros((kcount, bcount, mcount, fcount), dtype=dtype)
    print >> txt, 'Calculating phi-psi products'
    for kpt in kpt_u:
        # Note: will be reashaped to (n, i, m) like its name suggests
        overlaps_nim = np.zeros((bcount, mcount * fcount), dtype=dtype)
        lf.integrate(kpt.psit_nG, overlaps_nim, k=kpt.k)
        overlaps_nim.shape = (bcount, fcount, mcount)
        overlaps_knmi[kpt.u, :, :, :] = overlaps_nim.swapaxes(1, 2)

    print >> txt, 'Aligning matrices'
    for k in range(kcount):
        f_n = kpt_u[k].f_n
        # Apply weights depending on occupation
        for n in range(bcount):
            #    if n == bcount -1:
            #        w = 1.#f_n[n]
            #    else:
            #        w = 0.
            overlaps_knmi[k, n, :, :] *= f_n[n]
        
    S_kmii = np.zeros((kcount, mcount, fcount, fcount), dtype=dtype)
    conj_overlaps_knmi = overlaps_knmi.conjugate()

    for k in range(kcount):
        for m in range(mcount):
            for i in range(fcount):
                for j in range(fcount):
                    x1 = conj_overlaps_knmi[k, :, m, i]
                    x2 = overlaps_knmi[k, :, m, j]
                    S_kmii[k, m, i, j] = (x1 * x2).sum()

    assert s_kmii.shape == S_kmii.shape
    return s_kmii, S_kmii

def old_overlaps(l, gd, splines, kpt_u, center=(.5, .5, .5)):
    """Get scalar products of basis functions and references.

    Returns the triple-indexed matrices s and S, where::

        s    = < phi   | phi   > ,
         mij        mi      mj

              -----
               \     /        |  ~   \   /  ~   |        \ 
        S    =  )   (  phi    | psi   ) (  psi  | phi     )
         mij   /     \    mi  |    k /   \    k |     mj /
              -----
                k

    The functions phi are taken from the given splines, whereas psit
    must be on the grid represented by the GridDescriptor gd.
    Integrals are evaluated at the relative location given by center.
    """
    raise DeprecationWarning('Use overlaps method')
    # This method will be deleted, but presently we want to keep it
    # for testing
    assert len(kpt_u) == 1, 'This method only works for one k-point'
    kpt = kpt_u[0]
    psit_k = kpt.psit_nG

    mcounts = [spline.get_angular_momentum_number() for spline in splines]
    mcount = mcounts[0]
    for mcount_i in mcounts:
        assert mcount == mcount_i
    mcount = 2*l + 1
    fcount = len(splines)
    phi_lf = create_localized_functions(splines, gd, center)
    #print 'loc funcs boxes',len(phi_lf.box_b)
    
    phi_mi = gd.zeros(fcount * mcount) # one set for each phi
    coef_xi = np.identity(fcount * mcount)
    phi_lf.add(phi_mi, coef_xi)
    integrals = np.zeros((fcount * mcount, fcount * mcount))
    phi_lf.integrate(phi_mi, integrals)
    """Integral matrix contents (assuming l==1 so there are three m-values)

                --phi1--  --phi2--  --phi3-- ...
                m1 m2 m3  m1 m2 m3  m1 m2 m3 ...
               +---------------------------------
               |
         |   m1| x 0 0     x 0 0
        phi1 m2| 0 x 0     0 x 0   ...
         |   m3| 0 0 x     0 0 x 
               |
         |   m1|   .
        phi2 m2|   .
         |   m3|   .
             . |
             .

    We want < phi_mi | phi_mj >, and thus only the diagonal elements of
    each submatrix!  For l=1 the diagonal elements are all equal, but this
    is not true in general"""

    # phiproducts: for each m, < phi_mi | phi_mj >
    phiproducts_mij = np.zeros((mcount, fcount, fcount))
    for i in range(fcount):
        for j in range(fcount):
            ioff = mcount * i
            joff = mcount * j
            submatrix_ij = integrals[ioff:ioff + mcount,joff:joff + mcount]
            phiproducts_mij[:, i, j] = submatrix_ij.diagonal()
    # This should be ones in submatrix diagonals and zero elsewhere

    # Now calculate scalar products < phi_mi | psit_k >, where psit_k are
    # solutions from reference calculation
    psitcount = len(psit_k)
    integrals_kim = np.zeros((psitcount, fcount * mcount))
    phi_lf.integrate(psit_k, integrals_kim)

    # Now psiproducts[k] is a flat list, but we want it to be a matrix with
    # dimensions corresponding to f and m.
    # The first three elements correspond to the same localized function
    # and so on.
    # What we want is one whole matrix for each m-value.
    psiproducts_mik = np.zeros((mcount, fcount, psitcount))
    for m in range(mcount):
        for i in range(fcount):
            for k in range(psitcount):
                w = kpt.f_n[k] * kpt.weight
                psiproducts_mik[m, i, k] = w * integrals_kim[k, mcount * i + m]

    # s[mij] = < phi_mi | phi_mj >
    s = np.array([phiproducts_mij])

    # S[mij] = sum over k: < phi_mi | psit_k > < psit_k | phi_mj >
    S = np.array([[np.dot(psiproducts_ik, np.transpose(psiproducts_ik))
                    for psiproducts_ik in psiproducts_mik]])

    return s, S

def main():
    """Testing."""
    args = sys.argv[1:]
    if len(args) == 0:
        args = g2.atoms
    rcut = 6.
    generator = PolarizationOrbitalGenerator(rcut)
    import pylab
    for symbol in args:
        gd, psit_k, center = Reference(symbol, txt=None).get_reference_data()
        psitcount = len(psit_k)
        gramschmidt(gd, psit_k)
        print 'Wave function count', psitcount
        psit_k_norms = gd.integrate(psit_k * psit_k)

        Z, states = configurations[symbol]
        highest_state = states[-1]
        n, l_atom, occupation, energy = highest_state
        l = l_atom + 1
        
        phi = generator.generate(l, gd, psit_k, center, dtype=float)
        
        r = np.arange(0., rcut, .01)
        norm = get_norm(r, phi(r), l)

        quality = generator.quality
        orbital = 'spdf'[l]
        style = ['-.', '--','-',':'][l]
        pylab.plot(r, phi(r) * r**l, style,
                   label='%s [type=%s][q=%.03f]' % (symbol, orbital, quality))
    pylab.legend()
    pylab.show()

def dummy_kpt_test():
    l = 0
    rcut = 6.
    a = 5.
    k_kc = [(.5, .5, .5)]#[(0., 0., 0.), (0.5, 0.5, 0.5)]
    kcount = len(k_kc)
    dtype = complex
    r = np.arange(0., rcut, .01)

    spos_ac_ref = [(0., 0., 0.)]#, (.2, .2, .2)]
    spos_ac = [(0., 0., 0.), (.2, .2, .2)]


    ngaussians = 4
    realgaussindex = (ngaussians - 1) / 2

    rchars = np.linspace(1., rcut, ngaussians)
    splines = []
    gaussians = [QuasiGaussian(1./rch**2., rcut) for rch in rchars]
    for g in gaussians:
        norm = get_norm(r, g(r), l)
        g.renormalize(norm)
        spline = Spline(l, r[-1], g(r))
        splines.append(spline)

    refgauss = gaussians[realgaussindex]
    refspline = splines[realgaussindex]

    gd = GridDescriptor((60, 60, 60), (a,a,a), (1,1,1))

    reflf_a = [create_localized_functions([refspline], gd, spos_c, dtype=dtype)
               for spos_c in spos_ac_ref]
    for reflf in reflf_a:
        reflf.set_phase_factors(k_kc)

    kpt_u = [KPoint([], gd, 1., 0, k, k, k_c, dtype)
             for k, k_c in enumerate(k_kc)]
    
    for kpt in kpt_u:
        kpt.allocate(1)
        kpt.f_n[0] = 1.
        psit_nG = gd.zeros(1, dtype=dtype)
        coef_xi = np.identity(1, dtype=dtype)
        integral = np.zeros((1, 1), dtype=dtype)
        for reflf in reflf_a:
            reflf.add(psit_nG, coef_xi, k=kpt.k)
            reflf.integrate(psit_nG, integral, k=kpt.k)
        kpt.psit_nG = psit_nG
        print 'ref norm', integral

    print 'calculating overlaps'
    os_kmii, oS_kmii = overlaps(l, gd, splines, kpt_u,
                                spos_ac=spos_ac_ref)
    print 'done'

    lf_a = [create_localized_functions(splines, gd, spos_c, dtype=dtype)
            for spos_c in spos_ac]
    for lf in lf_a:
        lf.set_phase_factors(k_kc)

    s_kii = np.zeros((kcount, ngaussians, ngaussians), dtype=dtype)
    S_kii = np.zeros((kcount, ngaussians, ngaussians), dtype=dtype)

    for kpt in kpt_u:
        k = kpt.k
        all_integrals = np.zeros((1, ngaussians), dtype=dtype)
        tempgrids = gd.zeros(ngaussians, dtype=dtype)
        tempcoef_xi = np.identity(ngaussians, dtype=dtype)
        for lf in lf_a:
            lf.integrate(kpt.psit_nG, all_integrals, k=k)
            lf.add(tempgrids, tempcoef_xi, k=k)
            lf.integrate(tempgrids, s_kii[k], k=k)

        print 'all <phi|psi>'
        print all_integrals

        conj_integrals = np.conj(all_integrals)
        for i in range(ngaussians):
            for j in range(ngaussians):
                S_kii[k, i, j] = conj_integrals[0, i] * all_integrals[0, j]

    print 'handmade s_kmii'
    print s_kii

    print 'handmade S_ii'
    print S_kii

    s_kmii = s_kii.reshape(kcount, 1, ngaussians, ngaussians)
    S_kmii = S_kii.reshape(kcount, 1, ngaussians, ngaussians)

    print 'matrices from overlap function'
    print 's_kmii'
    print os_kmii
    print 'S_kmii'
    print oS_kmii

    optimizer = CoefficientOptimizer(s_kmii, S_kmii, ngaussians)
    coefficients = optimizer.find_coefficients()

    optimizer2 = CoefficientOptimizer(os_kmii, oS_kmii, ngaussians)
    coefficients2 = optimizer2.find_coefficients()

    print 'coefs'
    print coefficients
    print 'overlaps() coefs'
    print coefficients2
    print 'badness'
    print optimizer.evaluate(coefficients)
    exactsolution = [0.] * ngaussians
    exactsolution[realgaussindex] = 1.
    print 'badness of exact solution'
    print optimizer.evaluate(exactsolution)

    orbital = LinearCombination(coefficients, gaussians)
    orbital2 = LinearCombination(coefficients2, gaussians)
    norm = get_norm(r, orbital(r), l)
    norm2 = get_norm(r, orbital2(r), l)
    orbital.renormalize(norm)
    orbital2.renormalize(norm2)

    import pylab
    pylab.plot(r, refgauss(r), label='ref')
    pylab.plot(r, orbital(r), label='opt')
    pylab.plot(r, orbital2(r), '--', label='auto')
    pylab.legend()
    pylab.show()


def dummy_kpt_test2():
    l = 0
    rcut = 6.
    a = 5.
    k_c = (0.5,0.5,0.5)
    dtype=complex
    r = np.arange(0., rcut, .01)

    ngaussians = 8
    rchars = np.linspace(1., rcut/2., ngaussians + 1)[1:]
    print 'rchars',rchars
    rchar_ref = rchars[ngaussians // 2]
    print 'rchar ref',rchar_ref

    generator = PolarizationOrbitalGenerator(rcut, gaussians=rchars)

    # Set up reference system
    #alpha_ref = 1 / (rcut/4.) ** 2.
    alpha_ref = 1 / rchar_ref ** 2.
    ref = QuasiGaussian(alpha_ref, rcut)
    norm = get_norm(r, ref(r), l)
    ref.renormalize(norm)
    gd, kpt, center = make_dummy_kpt_reference(l, ref, k_c,
                                               rcut, a, 40, dtype)
    psit_nG = kpt.psit_nG
    kpt.f_n = np.array([1.])
    print 'Norm sqr', gd.integrate(psit_nG * psit_nG)
    #gramschmidt(gd, psit_nG)
    print 'Normalized norm sqr', gd.integrate(psit_nG * psit_nG)

    quasigaussians = [QuasiGaussian(1/rchar**2., rcut) for rchar in rchars]
    y = []
    for g in quasigaussians:
        norm = get_norm(r, g(r), l)
        g.renormalize(norm)
        y.append(g(r))
    splines = [Spline(l, rcut, f_g) for f_g in y]
    s_kmii, S_kmii = overlaps(l, gd, splines, [kpt],
                              spos_ac=[(.5, .5, .5)])

    orbital = generator.generate(l, gd, [kpt], [center], dtype=complex)
    print 'coefs'
    print np.array(orbital.coefs)

    print 'quality'
    print generator.qualities

    import pylab
    pylab.plot(r, ref(r), label='ref')
    pylab.plot(r, orbital(r), label='interp')
    pylab.legend()
    pylab.show()


def dummy_test(lmax=4, rcut=6., lmin=0): # fix args
    """Run a test using a Gaussian reference function."""
    dtype = complex
    generator = PolarizationOrbitalGenerator(rcut, gaussians=4)
    r = np.arange(0., rcut, .01)
    alpha_ref = 1. / (rcut/4.) ** 2.
    import pylab
    for l in range(lmin, lmax + 1):
        g = QuasiGaussian(alpha_ref, rcut)
        norm = get_norm(r, g(r), l)
        g.renormalize(norm)
        gd, psit_k, center, ref = make_dummy_reference(l, g, rcut,
                                                       dtype=dtype)
        k_kc = ((0.,0.,0.), (.5,.5,.5))
        kpt_u = [KPoint([], gd, 1., 0, i, i, k_c, dtype=dtype)
                 for i, k_c in enumerate(k_kc)]
        for kpt in kpt_u:
            kpt.allocate(1)
            kpt.f_n = np.array([2.])
            kpt.psit_nG = psit_k
        
        phi = generator.generate(l, gd, kpt_u, [center], dtype=dtype)
        
        pylab.figure(l)
        #pylab.plot(r, ref(r)*r**l, 'g', label='ref')
        pylab.plot(r, g(r)*r**l, 'b', label='g')
        pylab.plot(r, phi(r)*r**l, 'r--', label='pol')
        pylab.title('l=%d' % l)
        pylab.legend()
    pylab.show()

restart_filename = 'ref.%s.gpw'
output_filename = 'ref.%s.txt'

# XXX find a better way to do this
# Default characteristic radii when using only one gaussian

# Systems for non-dimer-forming or troublesome atoms
# 'symbol' : (g2 key, index of desired atom)

special_systems = {'Li' : ('LiF', 0),
                   'B' : ('BCl3', 0), # No boron dimer
                   'C' : ('CH4', 0), # No carbon dimer
                   'N' : ('NH3', 0), # double/triple bonds tend to be bad
                   'O' : ('H2O', 0), # O2 requires spin polarization
                   'F' : ('HF', 0),
                   'Na' : ('NaCl', 0),
                   'Al' : ('AlCl3', 0),
                   'Si' : ('SiO', 0), # No reason really.
                   'P' : ('PH3', 0),
                   'S' : ('SH2', 0), # S2 requires spin polarization
                   }

def get_system(symbol):
    """Get default reference formula or atomic index."""
    system = special_systems.get(symbol)
    if system is None:
        system = (symbol + '2', 0)
    return system

def get_systems(symbols=None):
    if symbols is None:
        symbols = g2.atoms
    systems = []
    for symbol in symbols:
        systems.append(get_system(symbol))
    return systems

class Reference:
    """Represents a reference function loaded from a file."""
    def __init__(self, symbol, filename=None, index=None, txt=None):
        if filename is None or filename == '-':
            formula, index = get_system(symbol)
            filename = restart_filename % formula
        calc = Calculator(filename, txt=txt)
        
        atoms = calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        if index is None:
            index = symbols.index(symbol)
        else:
            if not symbols[index] == symbol:
                raise ValueError(('Atom (%s) at specified index (%d) not of '+
                                  'requested type (%s)') % (symbols[index],
                                                            index, symbol))
        self.calc = calc
        self.filename = filename
        self.atoms = atoms
        self.symbol = symbol
        self.symbols = symbols
        self.index = index
        self.cell = atoms.get_cell().diagonal() # cubic cell
        #self.center = atoms.positions[index]
        self.spos_ac = atoms.positions / self.cell

        self.gpts = calc.wfs.gd.N_c
        if calc.kpt_u[0].psit_nG is None:
            raise RuntimeError('No wave functions found in .gpw file')

    def get_reference_data(self):
        c = self.calc
        for kpt in c.kpt_u:
            kpt.psit_nG = kpt.psit_nG[:] # load wave functions from the file
            # this is an ugly way to do it, by the way, but it probably works
        # Right now we only use one nuclear position, but maybe this
        # is to be changed in the future
        return c.wfs.gd, c.kpt_u, self.spos_ac[self.index:self.index+1]

if __name__ == '__main__':
    pass
