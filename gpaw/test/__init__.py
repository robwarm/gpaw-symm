import os
import gc
import sys
import time
import signal
import traceback

import numpy as np

from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.utilities import devnull, compiled_with_sl
from gpaw import setup_paths
from gpaw import mpi
import gpaw


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y).any() or (np.abs(x - y) > tolerance).any():
        msg = (msg + '%s != %s (error: |%s| > %.9g)' %
               (x, y, x - y, tolerance))
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)


def findpeak(x, y):
    dx = x[1] - x[0]
    i = y.argmax()
    a, b, c = np.polyfit([-1, 0, 1], y[i - 1:i + 2], 2)
    assert a < 0
    x = -0.5 * b / a
    return dx * (i + x), a * x**2 + b * x + c

    
def gen(symbol, exx=False, name=None, **kwargs):
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        g.run(exx=exx, name=name, use_restart_file=False, **parameters[symbol])
    mpi.world.barrier()
    if setup_paths[0] != '.':
        setup_paths.insert(0, '.')


def wrap_pylab(names=[]):
    """Use Agg backend and prevent windows from popping up."""
    import matplotlib
    matplotlib.use('Agg')
    import pylab

    def show(names=names):
        if names:
            name = names.pop(0)
        else:
            name = 'fig.png'
        pylab.savefig(name)

    pylab.show = show


tests = [
    'gemm_complex.py',
    'mpicomm.py',
    'ase3k_version.py',
    'numpy_core_multiarray_dot.py',
    'eigh.py',
    'lapack.py',
    'dot.py',
    'lxc_fxc.py',
    'blas.py',
    'erf.py',
    'gp2.py',
    'kptpar.py',
    'non_periodic.py',
    'parallel/blacsdist.py',
    'gradient.py',
    'cg2.py',
    'kpt.py',
    'lf.py',
    'gd.py',
    'parallel/compare.py',
    'pbe_pw91.py',
    'fsbt.py',
    'derivatives.py',
    'Gauss.py',
    'second_derivative.py',
    'integral4.py',
    'parallel/ut_parallel.py',
    'transformations.py',
    'parallel/parallel_eigh.py',
    'spectrum.py',
    'xc.py',
    'zher.py',
    'pbc.py',
    'lebedev.py',
    'parallel/ut_hsblacs.py',
    'parallel/submatrix_redist.py',
    'occupations.py',
    'dump_chi0.py',
    'cluster.py',
    'pw/interpol.py',
    'poisson.py',
    'pw/lfc.py',
    'pw/reallfc.py',
    'XC2.py',
    'multipoletest.py',
    'nabla.py',
    'noncollinear/xccorr.py',
    'gauss_wave.py',
    'harmonic.py',
    'atoms_too_close.py',
    'screened_poisson.py',
    'yukawa_radial.py',
    'noncollinear/xcgrid3d.py',
    'vdwradii.py',
    'lcao_restart.py',
    'ase3k.py',
    'parallel/ut_kptops.py',
    'fileio/idiotproof_setup.py',
    'fileio/hdf5_simple.py',
    'fileio/hdf5_noncontiguous.py',
    'fileio/parallel.py',
    'timing.py',
    'coulomb.py',
    'xcatom.py',
    'maxrss.py',
    'proton.py',
    'pw/moleculecg.py',
    'keep_htpsit.py',
    'pw/stresstest.py',
    'aeatom.py',
    'numpy_zdotc_graphite.py',
    'lcao_density.py',
    'parallel/overlap.py',
    'restart.py',
    # numpy/scipy tests fail randomly
    #'numpy_test.py',
    #'scipy_test.py',
    'gemv.py',
    'ylexpand.py',
    'potential.py',
    'wfs_io.py',
    'fixocc.py',
    'nonselfconsistentLDA.py',
    'gga_atom.py',
    'ds_beta.py',
    'gauss_func.py',
    'noncollinear/h.py',
    'symmetry.py',
    'symmetry_ft.py',
    'usesymm.py',
    'broydenmixer.py',
    'mixer.py',
    'pes.py',
    'wfs_auto.py',
    'ewald.py',
    'refine.py',
    'revPBE.py',
    'nonselfconsistent.py',
    'hydrogen.py',
    'fileio/file_reference.py',
    'fixdensity.py',
    'bee1.py',
    'spinFe3plus.py',
    'pw/h.py',
    'pw/fulldiag.py',
    'pw/fulldiagk.py',
    'stdout.py',
    'parallel/lcao_complicated.py',
    'pw/slab.py',
    'spinpol.py',
    'plt.py',
    'lcao_pair_and_coulomb.py',
    'eed.py',
    'lrtddft2.py',
    'parallel/hamiltonian.py',
    'pseudopotential/ah.py',
    'laplace.py',
    'pw/mgo_hybrids.py',
    'lcao_largecellforce.py',
    'restart2.py',
    'Cl_minus.py',
    'fileio/restart_density.py',
    'external_potential.py',
    'pw/bulk.py',
    'pw/fftmixer.py',
    'mgga_restart.py',
    'vdw/quick.py',
    'multipoleH2O.py',
    'bulk.py',
    'elf.py',
    'aluminum_EELS_RPA.py',
    'aluminum_EELS_ALDA.py',
    'H_force.py',
    'parallel/lcao_hamiltonian.py',
    'fermisplit.py',
    'parallel/ut_redist.py',
    'lcao_h2o.py',
    'cmrtest/cmr_test2.py',
    'h2o_xas.py',
    'ne_gllb.py',
    'exx_acdf.py',
    'asewannier.py',
    'exx_q.py',
    'ut_rsh.py',
    'ut_csh.py',
    'spin_contamination.py',
    'davidson.py',
    'partitioning.py',
    'pw/davidson_pw.py',
    'cg.py',
    'gllbatomic.py',
    'lcao_force.py',
    'neb.py',
    'fermilevel.py',
    'h2o_xas_recursion.py',
    'diamond_eps.py',
    'excited_state.py',
    # > 20 sec tests start here (add tests after gemm.py!)
    'gemm.py',
    'fractional_translations.py',
    'rpa_energy_Ni.py',
    'LDA_unstable.py',
    'si.py',
    'blocked_rmm_diis.py',
    'lxc_xcatom.py',
    'gw_planewave.py',
    'degeneracy.py',
    'apmb.py',
    'vdw/potential.py',
    'al_chain.py',
    'relax.py',
    'fixmom.py',
    'CH4.py',
    'diamond_absorption.py',
    'simple_stm.py',
    'gw_method.py',
    'lcao_bulk.py',
    'constant_electric_field.py',
    'parallel/ut_invops.py',
    'wannier_ethylene.py',
    'parallel/lcao_projections.py',
    'guc_force.py',
    'test_ibzqpt.py',
    'aedensity.py',
    'fd2lcao_restart.py',
    'gwsi.py',
    #'graphene_EELS.py', disabled while work is in progress on response code
    'lcao_bsse.py',
    'pplda.py',
    'revPBE_Li.py',
    'si_primitive.py',
    'complex.py',
    'Hubbard_U.py',
    'ldos.py',
    'parallel/ut_hsops.py',
    'pw/hyb.py',
    'pseudopotential/hgh_h2o.py',
    'vdw/quick_spin.py',
    'scfsic_h2.py',
    'lrtddft.py',
    'dscf_lcao.py',
    'IP_oxygen.py',
    'Al2_lrtddft.py',
    'rpa_energy_Si.py',
    '2Al.py',
    'tpss.py',
    'be_nltd_ip.py',
    'si_xas.py',
    'atomize.py',
    'chi0.py',
    'ralda_energy_H2.py',
    'ralda_energy_N2.py',
    'ralda_energy_Ni.py',
    'ralda_energy_Si.py',
    'Cu.py',
    'restart_band_structure.py',
    'ne_disc.py',
    'exx_coarse.py',
    'exx_unocc.py',
    'Hubbard_U_Zn.py',
    'muffintinpot.py',
    'diamond_gllb.py',
    'h2o_dks.py',
    'gw_ppa.py',
    'nscfsic.py',
    'gw_static.py',
    # > 100 sec tests start here (add tests after exx.py!)
    'response_na_plasmon.py',
    'exx.py',
    'pygga.py',
    'dipole.py',
    'nsc_MGGA.py',
    'mgga_sc.py',
    'MgO_exx_fd_vs_pw.py',
    'lb94.py',
    '8Si.py',
    'td_na2.py',
    'ehrenfest_nacl.py',
    'rpa_energy_N2.py',
    'beefvdw.py',
    #'mbeef.py',
    'nonlocalset.py',
    'wannierk.py',
    'rpa_energy_Na.py',
    'coreeig.py',
    'pw/si_stress.py',
    'ut_tddft.py',
    'transport.py',
    'vdw/ar2.py',
    'bse_sym.py',
    'aluminum_testcell.py',
    'au02_absorption.py',
    'lrtddft3.py',
    'scfsic_n2.py',
    'fractional_translations_big.py',
    'parallel/lcao_parallel.py',
    'parallel/lcao_parallel_kpt.py',
    'parallel/fd_parallel.py',
    'parallel/fd_parallel_kpt.py',
    'bse_aluminum.py',
    'bse_diamond.py',
    'bse_vs_lrtddft.py',
    'bse_silicon.py',
    'bse_MoS2_cut.py',
    'parallel/pblas.py',
    'parallel/scalapack.py',
    'parallel/scalapack_diag_simple.py',
    'parallel/scalapack_mpirecv_crash.py',
    'parallel/realspace_blacs.py',
    'AA_exx_enthalpy.py',
    #'usesymm2.py',
    #'eigh_perf.py', # Requires LAPACK 3.2.1 or later
    # XXX https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
    #'parallel/scalapack_pdlasrt_hang.py',
    #'dscf_forces.py',
    #'stark_shift.py',
    'cmrtest/cmr_test.py',
    'cmrtest/cmr_test3.py',
    'cmrtest/cmr_test4.py',
    'cmrtest/cmr_append.py',
    'cmrtest/Li2_atomize.py']

exclude = []

# not available on Windows
if os.name in ['ce', 'nt']:
    exclude += ['maxrss.py']

if mpi.size > 1:
    exclude += ['maxrss.py',
                'pes.py',
                'diamond_eps.py',
                'nscfsic.py',
                'coreeig.py',
                'asewannier.py',
                'wannier_ethylene.py',
                'muffintinpot.py',
                'stark_shift.py',
                'exx_q.py',
                'potential.py',
                #'cmrtest/cmr_test3.py',
                #'cmrtest/cmr_append.py',
                'cmrtest/Li2_atomize.py',  # started to hang May 2014
                'lcao_pair_and_coulomb.py',
                'bse_MoS2_cut.py',
                'pw/moleculecg.py',
                'pw/davidson_pw.py',
                # scipy.weave fails often in parallel due to
                # ~/.python*_compiled
                # https://github.com/scipy/scipy/issues/1895
                'scipy_test.py']

if mpi.size > 2:
    exclude += ['neb.py']

if mpi.size < 4:
    exclude += ['parallel/pblas.py',
                'parallel/scalapack.py',
                'parallel/scalapack_diag_simple.py',
                'parallel/realspace_blacs.py',
                'AA_exx_enthalpy.py',
                'bse_aluminum.py',
                'bse_diamond.py',
                'bse_silicon.py',
                'bse_vs_lrtddft.py',
                'fileio/parallel.py']

if mpi.size != 4:
    exclude += ['parallel/lcao_parallel.py']
    exclude += ['parallel/fd_parallel.py']
    exclude += ['parallel/scalapack_mpirecv_crash.py']
    exclude += ['parallel/scalapack_pdlasrt_hang.py']

if mpi.size == 1 or not compiled_with_sl():
    exclude += ['parallel/submatrix_redist.py']

if mpi.size != 1 and not compiled_with_sl():
    exclude += ['ralda_energy_H2.py',
                'ralda_energy_N2.py',
                'ralda_energy_Ni.py',
                'ralda_energy_Si.py',
                'bse_sym.py',
                'bse_silicon.py',
                'gwsi.py',
                'rpa_energy_N2.py',
                'pw/fulldiag.py',
                'pw/fulldiagk.py',
                'au02_absorption.py']

if mpi.size == 8:
    exclude += ['transport.py']

if mpi.size != 8:
    exclude += ['parallel/lcao_parallel_kpt.py']
    exclude += ['parallel/fd_parallel_kpt.py']

if sys.version_info < (2, 6):
    exclude.append('transport.py')
    
if np.__version__ < '1.6.0':
    exclude.append('chi0.py')
    
for test in exclude:
    if test in tests:
        tests.remove(test)


class TestRunner:
    def __init__(self, tests, stream=sys.__stdout__, jobs=1,
                 show_output=False):
        if mpi.size > 1:
            assert jobs == 1
        self.jobs = jobs
        self.show_output = show_output
        self.tests = tests
        self.failed = []
        self.skipped = []
        self.garbage = []
        if mpi.rank == 0:
            self.log = stream
        else:
            self.log = devnull
        self.n = max([len(test) for test in tests])

    def run(self):
        self.log.write('=' * 77 + '\n')
        if not self.show_output:
            sys.stdout = devnull
        ntests = len(self.tests)
        t0 = time.time()
        if self.jobs == 1:
            self.run_single()
        else:
            # Run several processes using fork:
            self.run_forked()

        sys.stdout = sys.__stdout__
        self.log.write('=' * 77 + '\n')
        self.log.write('Ran %d tests out of %d in %.1f seconds\n' %
                       (ntests - len(self.tests) - len(self.skipped),
                        ntests, time.time() - t0))
        self.log.write('Tests skipped: %d\n' % len(self.skipped))
        if self.failed:
            self.log.write('Tests failed: %d\n' % len(self.failed))
        else:
            self.log.write('All tests passed!\n')
        self.log.write('=' * 77 + '\n')
        return self.failed

    def run_single(self):
        while self.tests:
            test = self.tests.pop(0)
            try:
                self.run_one(test)
            except KeyboardInterrupt:
                self.tests.append(test)
                break

    def run_forked(self):
        j = 0
        pids = {}
        while self.tests or j > 0:
            if self.tests and j < self.jobs:
                test = self.tests.pop(0)
                pid = os.fork()
                if pid == 0:
                    exitcode = self.run_one(test)
                    os._exit(exitcode)
                else:
                    j += 1
                    pids[pid] = test
            else:
                try:
                    while True:
                        pid, exitcode = os.wait()
                        if pid in pids:
                            break
                except KeyboardInterrupt:
                    for pid, test in pids.items():
                        os.kill(pid, signal.SIGHUP)
                        self.write_result(test, 'STOPPED', time.time())
                        self.tests.append(test)
                    break
                if exitcode == 512:
                    self.failed.append(pids[pid])
                elif exitcode == 256:
                    self.skipped.append(pids[pid])
                del pids[pid]
                j -= 1

    def run_one(self, test):
        if self.jobs == 1:
            self.log.write('%*s' % (-self.n, test))
            self.log.flush()

        t0 = time.time()
        filename = gpaw.__path__[0] + '/test/' + test

        failed = False
        skip = False

        try:
            loc = {}
            execfile(filename, loc)
            loc.clear()
            del loc
            self.check_garbage()
        except KeyboardInterrupt:
            self.write_result(test, 'STOPPED', t0)
            raise
        except ImportError, ex:
            module = ex.args[0].split()[-1].split('.')[0]
            if module in ['scipy', 'cmr', '_gpaw_hdf5']:
                skip = True
            else:
                failed = True
        except Exception:
            failed = True

        mpi.ibarrier(timeout=60.0)  # guard against parallel hangs

        me = np.array(failed)
        everybody = np.empty(mpi.size, bool)
        mpi.world.all_gather(me, everybody)
        failed = everybody.any()
        skip = mpi.world.sum(int(skip))

        if failed:
            self.fail(test, np.argwhere(everybody).ravel(), t0)
            exitcode = 2
        elif skip:
            self.write_result(test, 'SKIPPED', t0)
            self.skipped.append(test)
            exitcode = 1
        else:
            self.write_result(test, 'OK', t0)
            exitcode = 0

        return exitcode

    def check_garbage(self):
        gc.collect()
        n = len(gc.garbage)
        self.garbage += gc.garbage
        del gc.garbage[:]
        assert n == 0, ('Leak: Uncollectable garbage (%d object%s) %s' %
                        (n, 's'[:n > 1], self.garbage))

    def fail(self, test, ranks, t0):
        if mpi.rank in ranks:
            if sys.version_info >= (2, 4, 0, 'final', 0):
                tb = traceback.format_exc()
            else:  # Python 2.3! XXX
                tb = ''
                traceback.print_exc()
        else:
            tb = ''
        if mpi.size == 1:
            text = 'FAILED!\n%s\n%s%s' % ('#' * 77, tb, '#' * 77)
            self.write_result(test, text, t0)
        else:
            tbs = {tb: [0]}
            for r in range(1, mpi.size):
                if mpi.rank == r:
                    mpi.send_string(tb, 0)
                elif mpi.rank == 0:
                    tb = mpi.receive_string(r)
                    if tb in tbs:
                        tbs[tb].append(r)
                    else:
                        tbs[tb] = [r]
            if mpi.rank == 0:
                text = ('FAILED! (rank %s)\n%s' %
                        (','.join([str(r) for r in ranks]), '#' * 77))
                for tb, ranks in tbs.items():
                    if tb:
                        text += ('\nRANK %s:\n' %
                                 ','.join([str(r) for r in ranks]))
                        text += '%s%s' % (tb, '#' * 77)
                self.write_result(test, text, t0)

        self.failed.append(test)

    def write_result(self, test, text, t0):
        t = time.time() - t0
        if self.jobs > 1:
            self.log.write('%*s' % (-self.n, test))
        self.log.write('%10.3f  %s\n' % (t, text))


if __name__ == '__main__':
    TestRunner(tests).run()
