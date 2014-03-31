from gpaw import GPAW

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(0.5 * k / nkpt, 0, 0) for k in range(nkpt + 1)]

calc = GPAW('si_664.gpw',
            txt='Si_harris.txt',
            kpts=kpts,
            fixdensity=True,
            nbands=20,
            usesymm=None,
            eigensolver='cg',
            convergence={'bands': 15})

calc.get_potential_energy()
calc.write('Si_harris.gpw')
