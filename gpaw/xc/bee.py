import numpy as np
from types import FloatType
from ase.parallel import rank

import _gpaw
from gpaw.xc import XC
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
from gpaw.xc.vdw import FFTVDWFunctional
from gpaw import debug


class BEE1(XCKernel):
    """GGA exchange expanded in a PBE-like basis."""
    def __init__(self, parameters=None):
        """BEE1.

        parameters : array
            [thetas,coefs] for the basis expansion.

        """

        if parameters is None:
            self.name = 'BEE1'
            parameters = [0.0, 1.0]
        else:
            self.name = 'BEE1?'
        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(18, parameters)
        self.type = 'GGA'


class BEE2(XCKernel):
    """GGA exchange expanded in Legendre polynomials."""
    def __init__(self, parameters=None):
        """BEE2.

        parameters: array
            [transformation,0.0,[orders],[coefs]].

        """

        if parameters is None:
            # LDA exchange
            t = [1.0, 0.0]
            coefs = [1.0]
            orders = [0.0]
            parameters = np.append(t, np.append(orders, coefs))
        else:
            assert len(parameters) > 2
            assert np.mod(len(parameters), 2) == 0
            assert parameters[1] == 0.0

        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(17, parameters)
        self.type = 'GGA'
        self.name = 'BEE2'


class BEEVDWKernel(XCKernel):
    """Kernel for BEEVDW functionals."""
    def __init__(self, bee, xcoefs, ldac, pbec):
        """BEEVDW kernel.

        parameters:

        bee : str
            choose BEE1 or BEE2 exchange basis expansion.
        xcoefs : array
            coefficients for exchange.
        ldac : float
            coefficient for LDA correlation.
        pbec : float
            coefficient for PBE correlation.

        """

        if bee is 'BEE1':
            self.BEE = BEE1(xcoefs)
        elif bee is 'BEE2':
            self.BEE = BEE2(xcoefs)
        else:
            raise ValueError('Unknown BEE exchange: %s', bee)

        self.LDAc = LibXC('LDA_C_PW')
        self.PBEc = LibXC('GGA_C_PBE')
        self.ldac = ldac
        self.pbec = pbec

        self.type = 'GGA'
        self.name = 'BEEVDW'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)

        self.BEE.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)

        e0_g = np.empty_like(e_g)
        dedn0_sg = np.empty_like(dedn_sg)
        dedsigma0_xg = np.empty_like(dedsigma_xg)
        for coef, kernel in [
            (self.ldac, self.LDAc),
            (self.pbec - 1.0, self.PBEc)]:
            dedn0_sg[:] = 0.0
            kernel.calculate(e0_g, n_sg, dedn0_sg, sigma_xg, dedsigma0_xg)
            e_g += coef * e0_g
            dedn_sg += coef * dedn0_sg
            if kernel.type == 'GGA':
                dedsigma_xg += coef * dedsigma0_xg


class BEEVDWFunctional(FFTVDWFunctional):
    """Base class for BEEVDW functionals."""
    def __init__(self, bee='BEE1', xcoefs=(0.0, 1.0),
                 ccoefs=(0.0, 1.0, 0.0), t=4.0, orders=None, Nr=2048,
                 **kwargs):
        """BEEVDW functionals.

        parameters:

        bee : str
            choose BEE1 or BEE2 exchange basis expansion.
        xcoefs : array-like
            coefficients for exchange.
        ccoefs : array-like
            LDA, PBE, nonlocal correlation coefficients
        t : float
            transformation for BEE2 exchange
        orders : array
            orders of Legendre polynomials for BEE2 exchange
        Nr : int
            Nr for FFT evaluation of vdW

        """

        if bee is 'BEE1':
            name = 'BEE1VDW'
            Zab = -0.8491
            soft_corr = False
        elif bee is 'BEE2':
            name = 'BEE2VDW'
            Zab = -1.887
            soft_corr = False
            if orders is None:
                orders = range(len(xcoefs))
            xcoefs = np.append([t, 0.0], np.append(orders, xcoefs))
        elif bee == 'BEEF-vdW':
            bee = 'BEE2'
            name = 'BEEF-vdW'
            Zab = -1.887
            soft_corr = True
            t, x, o, ccoefs = self.load_xc_pars('BEEF-vdW')
            xcoefs = np.append(t, np.append(o, x))
            self.t, self.x, self.o, self.c = t, x, o, ccoefs
            self.nl_type = 2
        else:
            raise KeyError('Unknown BEEVDW functional: %s', bee)

        assert isinstance(Nr, int)
        assert Nr % 512 == 0

        ldac, pbec, vdw = ccoefs
        kernel = BEEVDWKernel(bee, xcoefs, ldac, pbec)
        FFTVDWFunctional.__init__(self, name=name, soft_correction=soft_corr,
                                  kernel=kernel, Zab=Zab, vdwcoef=vdw, Nr=Nr,
                                  **kwargs)

    def get_setup_name(self):
        return 'PBE'

    def load_xc_pars(self, name):
        """Get BEEF-vdW parameters"""
        assert name == 'BEEF-vdW'

        t = np.array([4.0, 0.0])
        c = np.array([ 0.600166476948828631066,
                       0.399833523051171368934,
                       1.0])
        x = np.array([ 1.516501714304992365356,
                       0.441353209874497942611,
                      -0.091821352411060291887,
                      -0.023527543314744041314,
                       0.034188284548603550816,
                       0.002411870075717384172,
                      -0.014163813515916020766,
                       0.000697589558149178113,
                       0.009859205136982565273,
                      -0.006737855050935187551,
                      -0.001573330824338589097,
                       0.005036146253345903309,
                      -0.002569472452841069059,
                      -0.000987495397608761146,
                       0.002033722894696920677,
                      -0.000801871884834044583,
                      -0.000668807872347525591,
                       0.001030936331268264214,
                      -0.000367383865990214423,
                      -0.000421363539352619543,
                       0.000576160799160517858,
                      -0.000083465037349510408,
                      -0.000445844758523195788,
                       0.000460129009232047457,
                      -0.000005231775398304339,
                      -0.000423957047149510404,
                       0.000375019067938866537,
                       0.000021149381251344578,
                      -0.000190491156503997170,
                       0.000073843624209823442])
        o = range(len(x))
        return t, x, o, c


class BEEF_Ensemble:
    """BEEF ensemble error estimation."""
    def __init__(self, calc=None, exch=None, corr=None):
        """BEEF ensemble

        parameters:

        calc : object
            Calculator holding a selfconsistent BEEF type electron density.
            May be BEEF-vdW or mBEEF.
        exch : array
            Exchange basis function contributions to the total energy.
            Defaults to None.
        corr : array
            Correlation basis function contributions to the total energy.
            Defaults to None.

        """

        self.calc = calc
        self.exch = exch
        self.corr = corr
        self.e_dft = None
        self.e0 = None
        if self.calc is None:
            raise KeyError('calculator not specified')

        # determine functional and read parameters
        self.xc = self.calc.get_xc_functional()
        if self.xc in ['BEEF-vdW', 'BEEF-1']:
            self.bee = BEEVDWFunctional('BEEF-vdW')
            self.bee_type = 1
            self.nl_type = self.bee.nl_type
            self.t = self.bee.t
            self.x = self.bee.x
            self.o = self.bee.o
            self.c = self.bee.c
        elif self.xc == 'mBEEF':
            self.bee = LibXC('mBEEF')
            self.bee_type = 2
            self.max_order = 8
            self.trans = [6.5124, -1.0]
            if self.exch is None and rank == 0:
                self.calc.converge_wave_functions()
                print 'wave functions converged'
        else:
            raise NotImplementedError('xc = %s not implemented' % self.xc)

    def create_xc_contributions(self, type):
        """General function for creating exchange or correlation energies"""
        assert type in ['exch', 'corr']
        err = 'bee_type %i not implemented' % self.bee_type

        if type == 'exch':
            if self.bee_type == 1:
                out = self.beefvdw_energy_contribs_x()
            elif self.bee_type == 2:
                out = self.mbeef_exchange_energy_contribs()
            else:
                raise NotImplementedError(err)
        else:
            if self.bee_type == 1:
                out = self.beefvdw_energy_contribs_c()
            elif self.bee_type == 2:
                out = np.array([])
            else:
                raise NotImplementedError(err)
        return out

    def get_non_xc_total_energies(self):
        """Compile non-XC total energy contributions"""
        if self.e_dft is None:
            self.e_dft = self.calc.get_potential_energy()
        if self.e0 is None:
            from gpaw.xc.kernel import XCNull
            xc_null = XC(XCNull())
            self.e0 = self.e_dft + self.calc.get_xc_difference(xc_null)
        isinstance(self.e_dft, FloatType)
        isinstance(self.e0, FloatType)

    def mbeef_exchange_energy_contribs(self):
        """Legendre polynomial exchange contributions to mBEEF Etot"""
        self.get_non_xc_total_energies()
        e_x = np.zeros((self.max_order, self.max_order))
        for p1 in range(self.max_order):  # alpha
            for p2 in range(self.max_order):  # s2
                pars_i = np.array([1, self.trans[0], p2, 1.0])
                pars_j = np.array([1, self.trans[1], p1, 1.0])
                pars = np.hstack((pars_i, pars_j))
                x = XC('2D-MGGA', pars)
                e_x[p1, p2] = self.e_dft + self.calc.get_xc_difference(x) - self.e0
                del x
        return e_x

    def beefvdw_energy_contribs_x(self):
        """Legendre polynomial exchange contributions to BEEF-vdW Etot"""
        self.get_non_xc_total_energies()
        e_pbe = self.e_dft + self.calc.get_xc_difference('GGA_C_PBE') - self.e0

        exch = np.zeros(len(self.o))
        for p in self.o:
            pars = [self.t[0], self.t[1], p, 1.0]
            bee = XC('BEE2', pars)
            exch[p] = self.e_dft + self.calc.get_xc_difference(bee) - self.e0 - e_pbe
            del bee
        return exch

    def beefvdw_energy_contribs_c(self):
        """LDA and PBE correlation contributions to BEEF-vdW Etot"""
        self.get_non_xc_total_energies()
        e_lda = self.e_dft + self.calc.get_xc_difference('LDA_C_PW') - self.e0
        e_pbe = self.e_dft + self.calc.get_xc_difference('GGA_C_PBE') - self.e0
        corr = np.array([e_lda, e_pbe])
        return corr
