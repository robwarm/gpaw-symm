import numpy as np
from pylab import *

A = loadtxt('frequency_equidistant.dat').transpose()
plot(A[0], A[1], label='Equidistant')

xlabel('Frequency [eV]', fontsize=18)
ylabel('Integrand', fontsize=18)
axis([0, 50, None, None])
legend(loc='lower right')
#show()
savefig('E_w.png')
