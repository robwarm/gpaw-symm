extra_compile_args = ['-std=c99', '-O3']
compiler = 'cc'
mpicompiler = 'cc'
mpilinker= 'cc'
libraries = []
extra_link_args += ['-dynamic']

scalapack = True
hdf5 = True

define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]
