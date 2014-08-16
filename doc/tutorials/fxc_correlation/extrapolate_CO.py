from ase.utils.extrapolate import extrapolate
import numpy as np

a = np.loadtxt('rpa_CO.dat')
ext, A, B, sigma = extrapolate(a[:,0], a[:,1], reg=3, plot=False)

a = np.loadtxt('rapbe_CO.dat')
ext, A, B, sigma = extrapolate(a[:,0], a[:,1], reg=3, plot=False)
