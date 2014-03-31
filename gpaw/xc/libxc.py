import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw import debug

short_names = {
    'LDA':     'LDA_X+LDA_C_PW',
    'PW91':    'GGA_X_PW91+GGA_C_PW91',
    'PBE':     'GGA_X_PBE+GGA_C_PBE',
    'PBEsol':  'GGA_X_PBE_SOL+GGA_C_PBE_SOL',
    'revPBE':  'GGA_X_PBE_R+GGA_C_PBE',
    'RPBE':    'GGA_X_RPBE+GGA_C_PBE',
    'BLYP':    'GGA_X_B88+GGA_C_LYP',
    'HCTH407': 'GGA_XC_HCTH_407',
    'WC':      'GGA_X_WC+GGA_C_PBE',
    'AM05':    'GGA_X_AM05+GGA_C_AM05',
#    'TPSSLXC':    'MGGA_X_TPSS+MGGA_C_TPSS',
    'M06LLXC':    'MGGA_X_M06_L+MGGA_C_M06_L',
#    'revTPSSLXC': 'MGGA_X_REVTPSS+MGGA_C_REVTPSS',
#    'oTPSS':   'MGGA_X_OTPSS+MGGA_C_OTPSS',
#    'MS0':     'MGGA_X_MS0+GGA_C_VPBE',
#    'MS1':     'MGGA_X_MS1+GGA_C_VPBE',
#    'MS2':     'MGGA_X_MS2+GGA_C_VPBE',
#    'mBEEF':   'MGGA_X_MBEEF+GGA_C_PBE_SOL'
}

class LibXC(XCKernel):
    def __init__(self, name):
        self.name = name
        self.initialize(nspins=1)

    def initialize(self, nspins):
        self.nspins = nspins
        name = short_names.get(self.name, self.name)
        number = _gpaw.lxcXCFuncNum(name)
        if number is not None:
            f = number
            xc = -1
            x = -1
            c = -1
            if '_XC_' in name:
                xc = f
            elif '_C_' in name:
                c = f
            else:
                x = f
        else:
            try:
                x, c = name.split('+')
            except ValueError:
                raise NameError('Unknown functional: "%s".' % name)
            xc = -1
            x = _gpaw.lxcXCFuncNum(x)
            c = _gpaw.lxcXCFuncNum(c)

        self.xc = _gpaw.lxcXCFunctional(xc, x, c, nspins)

        if self.xc.is_mgga():
            self.type = 'MGGA'
        elif self.xc.is_gga():
            self.type = 'GGA'
        else:
            self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        self.xc.calculate(e_g.ravel(), n_sg, dedn_sg,
                          sigma_xg, dedsigma_xg,
                          tau_sg, dedtau_sg)
