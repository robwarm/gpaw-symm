import numpy as np
from pylab import *

A = loadtxt('con_freq.dat').transpose()
plot(A[0], A[1], 'o-')

xlabel('Number of frequence points', fontsize=18)
ylabel('Energy', fontsize=18)
axis([None, None, -14.9, -14.2])
#show()
savefig('con_freq.png')
