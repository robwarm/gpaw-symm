#User provided customizations for the gpaw setup

#Here, one can override the default arguments, or append own
#arguments to default ones
#To override use the form
#     libraries = ['somelib','otherlib']
#To append use the form
#     libraries += ['somelib','otherlib']

compiler = 'icc.py'
#libraries += []
#libraries += ['mkl_intel_lp64','mkl_sequential','mkl_core', 'mkl_lapack', 'pthread','m']
#libraries += ['mkl_intel_lp64','mkl_sequential','mkl_core', 'pthread','m']

#libraries += ['mkl_intel_lp64' ,'mkl_sequential' ,'mkl_core','mkl_lapack95_lp64','pthread']
#libraries += ['mkl_rt']
libraries  = ['fftw3']
libraries += ['mkl_intel_lp64', 'mkl_core', 'mkl_sequential', 'pthread', 'm']

#library_dirs = []
library_dirs += ['/home/rbw/programs/installs/openmpi-1.8.1-intel2013.1.3/lib64']
library_dirs += ['/opt/intel/mkl/lib/intel64']
library_dirs += ['/home/rbw/programs/installs/fftw-3.3.4-intel2013.1.3/lib64']

#include_dirs = []
include_dirs += ['/home/rbw/programs/installs/openmpi-1.8.1-intel2013.1.3/include']
include_dirs += ['-I/opt/intel/mkl/include']
include_dirs += ['/home/rbw/programs/installs/fftw-3.3.4-intel2013.1.3/include']

#extra_link_args += []
extra_link_args += ["-fPIC",'-lpthread','-lm']

#extra_compile_args += []
#extra_compile_args += ['-I/opt/intel/mkl/include','-O1','-ipo','-no-prec-div','-static','-std=c99','-fPIC']
#extra_compile_args += ['-I/opt/intel/mkl/include','-mkl=sequential','-static','-std=c99','-fPIC']
extra_compile_args += ['-I/opt/intel/mkl/include','-static','-std=c99','-fPIC']

#runtime_library_dirs = []
#runtime_library_dirs += []

#extra_objects = []
#extra_objects += []

#define_macros = []
#define_macros += []

mpicompiler = 'icc.py'
mpilinker = 'icc.py'
#mpi_libraries = []
#mpi_libraries += []

#mpi_library_dirs = []
#mpi_library_dirs += []

#mpi_include_dirs = []
#mpi_include_dirs += []

#mpi_runtime_library_dirs = []
#mpi_runtime_library_dirs += []

#mpi_define_macros = []
#mpi_define_macros += []

#platform_id = ''

hdf5 = False

# Valid values for scalapack are False, or True:
# False (the default) - no ScaLapack compiled in
# True - ScaLapack compiled in
# Warning! At least scalapack 2.0.1 is required!
# See https://trac.fysik.dtu.dk/projects/gpaw/ticket/230
scalapack = False

if scalapack:
  #  libraries += ['scalapack']
  #  library_dirs += ['/opt/intel/mkl/lib/intel64']
    define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1')]
    define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
    
define_macros += [("GPAW_ASYNC",1)]
define_macros += [("GPAW_MPI2",1)]


# In order to link libxc installed in a non-standard location
# (e.g.: configure --prefix=/home/user/libxc-2.0.1-1), use:
# - static linking:
#include_dirs += ['/home/rbw/programs/installs/libxc-2.0.1-intel2013.5/include']
#extra_link_args += ['/home/rbw/programs/installs/libxc-2.0.1-intel2013.5/lib64/libxc.a']
#if 'xc' in libraries: libraries.remove('xc')
# - dynamic linking (requires also setting LD_LIBRARY_PATH at runtime):
include_dirs += ['/home/rbw/programs/installs/libxc-2.2.0-intel2013.1.3/include']
library_dirs += ['/home/rbw/programs/installs/libxc-2.2.0-intel2013.1.3/lib64']
if 'xc' not in libraries: libraries.append('xc')
