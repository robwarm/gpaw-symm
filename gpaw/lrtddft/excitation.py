"""Excitation lists base classes

"""
from math import sqrt

import numpy as np

import gpaw.mpi as mpi
from gpaw.output import initialize_text_stream
from ase.units import A, m, s, Bohr, _aut, C

class ExcitationList(list):
    """General Excitation List class.

    """
    def __init__(self, calculator=None, txt=None):

        # initialise empty list
        list.__init__(self)

        self.calculator = calculator
        if not txt and calculator:
            txt = calculator.txt
        self.txt, firsttime = initialize_text_stream(txt, mpi.rank)

    def get_calculator(self):
        return self.calculator

    def get_energies(self):
        """Get excitation energies in Hartrees"""
        el = []
        for ex in self:
            el.append(ex.get_energy())
        return np.array(el)

    def get_trk(self):
        """Evaluate the Thomas Reiche Kuhn sum rule"""
        trkm = np.zeros((3))
        for ex in self:
            trkm += ex.get_energy() * ex.get_dipol_me()**2
        return 2. * trkm # scale to get the number of electrons XXX spinpol ?
    
    def get_polarizabilities(self, lmax=7):
        """Calculate the Polarisabilities
        see Jamorski et al. J. Chem. Phys. 104 (1996) 5134"""
        S = np.zeros((lmax+1))
        for ex in self:
            e = ex.get_energy()
            f = ex.get_oscillator_strength()[0]
            for l in range(lmax+1):
                S[l] += e**(-2 * l) * f
        return S

    def set_calculator(self, calculator):
        self.calculator = calculator

    def __str__(self):
        string = '# ' + str(type(self))
        if len(self) != 0:
            string += ', %d excitations:' % len(self)
        string += '\n'
        for ex in self:
            string += '#  '+ex.__str__()+"\n"
        return string
        
    def get_alpha(self, omega):
        """Return the polarization tensor"""

        alpha_cc = np.zeros((3,3))
        for ex in self:
            alpha_cc += ex.get_alpha(omega)
        return alpha_cc

class Excitation:
    def get_energy(self):
        """Get the excitations energy relative to the ground state energy
        in Hartrees.
        """
        return self.energy
    
    def get_dipol_me(self):
        """return the excitations dipole matrix element
        including the occupation factor"""
        return self.me / sqrt(self.energy)
    
    def get_oscillator_strength(self, form='r'):
        """Return the excitations dipole oscillator strength.


        self.me is assumed to be::

          form='r': sqrt(f * E) * <I|r|J>,
          form='v': sqrt(f / E) * <I|d/(dr)|J>

        for f = multiplicity, E = transition energy and initial and
        final states::
        
          |I>, |J>
          
        """
        
        if form == 'r':
            # length form
            me = self.me
        elif form == 'v':
            # velocity form
            me = self.muv * np.sqrt(self.energy)
        else:
            raise RuntimeError('Unknown form >' + form + '<')

        osz = [0.]
        for c in range(3):
            val = 2. * me[c]**2
            osz.append(val)
            osz[0] += val / 3.
        
        return osz

    def get_rotatory_strength(self, form='r', units='cgs'):
        """Return rotatory strength"""
        if self.magn is None:
            raise RuntimeError('Magnetic moment not available.')

        if units =='cgs':
            # 10^-40 esu cm erg / G
            # = 3.33564095 * 10^-15 A^2 m^3 s
            # conversion factor after
            # T. B. Pedersen and A. E. Hansen, 
            # Chem. Phys. Lett. 246 (1995) 1
            # pre = 471.43
            # From TurboMole
            pre = 64604.8164
        elif uints == 'a.u.':
            pre = 1.
        else:
            raise RuntimeError('Unknown units >' + units + '<')

        if form == 'r':
            # length form
            mu = self.mur
        elif form == 'v':
            # velocity form
            mu = self.muv
        else:
            raise RuntimeError('Unknown form >' + form + '<')
        
        return pre * np.dot(mu, self.magn)
        
    def set_energy(self, E):
        """Set the excitations energy relative to the ground state energy"""
        self.energy = E
    
    def get_alpha(self, omega):
        """Return the polarization tensor"""
        me = self.me

        alpha_cc = np.zeros((3,3))
        for c1 in range(3):
            for c2 in range(c1, 3):
                alpha_cc[c1, c2] = alpha_cc[c2, c1] = me[c1] * me[c2]
                
        return 2 * self.energy / (self.energy**2 - omega**2) * alpha_cc

