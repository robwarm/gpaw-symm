import numpy as np

from ase.units import Bohr
from ase.lattice import bulk
from ase.structure import molecule

from gpaw import GPAW
from gpaw.response.df import DF
from gpaw.mpi import serial_comm
from gpaw.wavefunctions.pw import PWLFC
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC


for mode in ('fd', 'pw'):
    for sys in ('atoms', 'solids', 'solidswithspin'):
        spinpol = False
        if sys == 'atoms':
            atoms = molecule('O2')
            atoms.set_pbc(True)
            atoms.set_cell([4.0, 4.0, 5.0], scale_atoms=False)
            atoms.center()
            kpts = (1, 1, 1)
            spinpol = True
        elif sys == 'solids':
            atoms = bulk('Si', 'fcc', a=3.25)
            kpts = (2, 2, 2)
        elif sys == 'solidswithspin':
            atoms = bulk('Ni', 'bcc', a=2.88)
            atoms[0].magmom = 0.7
            kpts = (2, 2, 2)
            spinpol = True
            
        if mode == 'fd':
            calc = GPAW(h=0.18, kpts=kpts, basis='dzp', maxiter=500,
                        usesymm=None,
                        communicator=serial_comm,
                        spinpol=spinpol)
        
        else:
            calc = GPAW(mode='pw', kpts=kpts, basis='dzp', dtype=complex,
                        maxiter=500,
                        usesymm=None, communicator=serial_comm)

        atoms.set_calculator(calc)
        atoms.get_potential_energy()

        gd = GridDescriptor(calc.wfs.gd.N_c, calc.atoms.cell / Bohr,
                            pbc_c=True, comm=serial_comm)
        kd = calc.wfs.kd
        setups = calc.wfs.setups
        bzk_kc = calc.wfs.kd.bzk_kc
        spos_ac = calc.atoms.get_scaled_positions()
        nbands = calc.get_number_of_bands()
        nspins = calc.wfs.nspins
        df = DF(calc)
        df.spos_ac = spos_ac
        
        if mode == 'fd':
            pt = LFC(gd, [setup.pt_j for setup in setups],
                              KPointDescriptor(bzk_kc),
                              dtype=calc.wfs.dtype)
            pt.set_positions(spos_ac)
        
            for spin in range(nspins):
                for k in range(len(bzk_kc)):
                    ibzk = k  # since no symmetry
                    u = kd.get_rank_and_index(spin, ibzk)[1]
                    kpt = calc.wfs.kpt_u[u]
                    for n in range(nbands):
                        P_ai = pt.dict()
                        psit_G = calc.wfs.get_wave_function_array(n, ibzk,
                                                                  spin)
                        pt.integrate(psit_G, P_ai, ibzk)
                            
                        for a in range(len(P_ai)):
                            assert np.abs(
                                P_ai[a] -
                                calc.wfs.kpt_u[u].P_ani[a][n]).sum() < 1e-8
                            assert np.abs(
                                P_ai[a] -
                                df.get_P_ai(k, n, spin)[a]).sum() < 1e-8
        
        else:
            pt = PWLFC([setup.pt_j for setup in setups], calc.wfs.pd)
            pt.set_positions(spos_ac)
        
            for spin in range(nspins):
                for k in range(len(bzk_kc)):
                    ibzk = k  # since no symmetry
                    u = kd.get_rank_and_index(spin, ibzk)[1]
                    kpt = calc.wfs.kpt_u[u]
                    for n in range(nbands):
                        Ptmp_ai = pt.dict()
                        # here psit_G is planewave coefficient:
                        pt.integrate(kpt.psit_nG[n], Ptmp_ai, ibzk)
                        P_ai = df.get_P_ai(k, n, spin, Ptmp_ai)
                        for a in range(len(P_ai)):
                            assert np.abs(
                                P_ai[a] -
                                calc.wfs.kpt_u[u].P_ani[a][n]).sum() < 1e-8
