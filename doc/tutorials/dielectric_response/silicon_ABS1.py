import numpy as np
from gpaw import GPAW
from gpaw.response.df import DF

w = np.linspace(0, 24., 481)    # 0-24 eV with 0.05 eV spacing
q = np.array([0.0, 0.00001, 0.])

df = DF(calc='si.gpw',
        q=q,
        w=w,
        eta=0.1,           # Broadening parameter 
        ecut=150,          # Energy cutoff for planewaves
        optical_limit=True,
        txt='df_2.out')    # Output text

df.get_absorption_spectrum(filename='si_abs.dat')
df.check_sum_rule()
df.write('df_2.pckl')      # Save important parameters and data 
