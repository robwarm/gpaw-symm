# Copyright (C) 2006 CSC-Scientific Computing Ltd.

# Please see the accompanying LICENSE file for further information.

import os
import platform
import sys
import re
import distutils.util
from distutils.sysconfig import get_config_var, get_config_vars
from distutils.command.config import config
from glob import glob
from os.path import join
from stat import ST_MTIME

def check_packages(packages, msg, include_ase, import_numpy):
    """Check the python version and required extra packages

    If ASE is not installed, the `packages` list is extended with the
    ASE modules if they are found."""

    if sys.version_info < (2, 3, 0, 'final', 0):
        raise SystemExit('Python 2.3.1 or later is required!')

    if import_numpy:
        try:
            import numpy
        except ImportError:
            raise SystemExit('numpy is not installed!')
    else:
        msg += ['* numpy is not installed.',
                '  "include_dirs" in your customize.py must point to "numpy/core/include".']

    if not include_ase:
        if import_numpy:
            try:
                import ase
            except ImportError:
                import_ase = True
            else:
                import_ase = False
        else:
            import_ase = False

    if include_ase or import_ase:
        # Find ASE directories:
        # include_ase works in case:
        # cd gpaw # top-level gpaw source directory
        # tar zxf ~/python-ase-3.1.0.846.tar.gz
        # ln -s python-ase-3.1.0.846/ase .
        ase_root = 'ase'
        if include_ase:
            assert os.path.isdir(ase_root), ase_root+': No such file or directory'
        ase = []
        for root, dirs, files in os.walk(ase_root):
            if 'CVS' in dirs:
                dirs.remove('CVS')
            if '.svn' in dirs:
                dirs.remove('.svn')
            if '__init__.py' in files:
                ase.append(root.replace('/', '.'))

        if len(ase) == 0:
            msg += ['* ASE is not installed!  You may be able to install',
                    "  gpaw, but you can't use it without ASE!"]
        else:
            packages += ase

def find_file(arg, dir, files):
    #looks if the first element of the list arg is contained in the list files
    # and if so, appends dir to to arg. To be used with the os.path.walk
    if arg[0] in files:
        arg.append(dir)


def get_system_config(define_macros, undef_macros,
                      include_dirs, libraries, library_dirs, extra_link_args,
                      extra_compile_args, runtime_library_dirs, extra_objects,
                      msg, import_numpy):

    undef_macros += ['NDEBUG']
    if import_numpy:
        import numpy
        include_dirs += [numpy.get_include()]

    # libxc
    libraries += ['xc']

    machine = platform.uname()[4]
    if machine == 'sun4u':

        #  _
        # |_ | ||\ |
        #  _||_|| \|
        #

        extra_compile_args += ['-Kpic', '-fast']

        # Suppress warning from -fast (-xarch=native):
        f = open('cc-test.c', 'w')
        f.write('int main(){}\n')
        f.close()
        stderr = os.popen3('cc cc-test.c -fast')[2].read()
        arch = re.findall('-xarch=(\S+)', stderr)
        os.remove('cc-test.c')
        if len(arch) > 0:
            extra_compile_args += ['-xarch=%s' % arch[-1]]


        # We need the -Bstatic before the -lsunperf and -lfsu:
        # http://forum.java.sun.com/thread.jspa?threadID=5072537&messageID=9265782
        extra_link_args += ['-Bstatic', '-lsunperf', '-lfsu', '-Bdynamic']
        cc_version = os.popen3('cc -V')[2].readline().split()[3]
        if cc_version > '5.6':
            libraries.append('mtsk')
        else:
            extra_link_args.append('-lmtsk')
        #define_macros.append(('NO_C99_COMPLEX', '1'))

        msg += ['* Using SUN high performance library']

    elif sys.platform.startswith('win'):

        # We compile with mingw coming from pythonyx (32-bit)
        # on the msys command line, e.g.:
        # LIBRARY_PATH=/c/libxc/lib:/c/OpenBLAS/lib \
        # C_INCLUDE_PATH=/c/libxc/include python setup.py build
        if 'LIBRARY_PATH' in os.environ:
            library_dirs += os.environ['LIBRARY_PATH'].split(os.path.pathsep)

        extra_compile_args += ['-Wall', '-std=c99']

        lib = ''
        for ld in library_dirs:
            # OpenBLAS (includes Lapack)
            if os.path.exists(join(ld, 'libopenblas.a')):
                lib = 'openblas'
                directory = ld
                break
        if lib == 'openblas':
             libraries += [lib, 'gfortran']
        if lib:
             msg +=  ['* Using %s library from %s' % (lib, directory)]

    elif sys.platform in ['aix5', 'aix6']:

        #
        # o|_  _ _
        # ||_)| | |
        #

        extra_compile_args += ['-qlanglvl=stdc99']
        # setting memory limit is necessary on aix5
        if sys.platform == 'aix5':
            extra_link_args += ['-bmaxdata:0x80000000',
                '-bmaxstack:0x80000000']

        libraries += ['f', 'lapack', 'essl']
        define_macros.append(('GPAW_AIX', '1'))

    elif machine == 'x86_64':

        #    _
        # \/|_||_    |_ |_|
        # /\|_||_| _ |_|  |
        #

        extra_compile_args += ['-Wall', '-std=c99']

        # Look for ACML libraries:
        acml = glob('/opt/acml*/g*64/lib')
        if len(acml) > 0:
            library_dirs += [acml[-1]]
            libraries += ['acml']
            if acml[-1].find('gfortran') != -1: libraries.append('gfortran')
            if acml[-1].find('gnu') != -1: libraries.append('g2c')
            extra_link_args += ['-Wl,-rpath=' + acml[-1]]
            msg += ['* Using ACML library']
        else:
            atlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib64/atlas']:
                if glob(join(dir, 'libatlas.so')) != []:
                    atlas = True
                    libdir = dir        
                    break
            openblas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib64']:
                if glob(join(dir, 'libopenblas.so')) != []:
                    openblas = True
                    libdir = dir        
                    break
            if openblas:  # prefer openblas
                libraries += ['openblas', 'lapack']
                library_dirs += [libdir]
                msg +=  ['* Using OpenBLAS library']
            else:
                if atlas:  # then atlas
                    # http://math-atlas.sourceforge.net/errata.html#LINK
                    # atlas does not respect OMP_NUM_THREADS - build single-thread
                    # http://math-atlas.sourceforge.net/faq.html#tsafe
                    libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                    library_dirs += [libdir]
                    msg +=  ['* Using ATLAS library']
                else:
                    libraries += ['blas', 'lapack']
                    msg +=  ['* Using standard lapack']

    elif machine =='ia64':

        #  _  _
        # |_ |  o
        #  _||_||
        #

        extra_compile_args += ['-Wall', '-std=c99']
        libraries += ['mkl','mkl_lapack64']

    elif machine == 'i686':

        #      _
        # o|_ |_||_
        # ||_||_||_|
        #

        extra_compile_args += ['-Wall', '-std=c99']

        if 'MKL_ROOT' in os.environ:
            mklbasedir = [os.environ['MKL_ROOT']]
        else:
            mklbasedir = glob('/opt/intel/mkl*')
        libs = ['libmkl_ia32.a']
        if mklbasedir != []:
            os.path.walk(mklbasedir[0],find_file, libs)
        libs.pop(0)
        if libs != []:
            libs.sort()
            libraries += ['mkl_lapack',
                          'mkl_ia32', 'guide', 'pthread', 'mkl']#, 'mkl_def']
            library_dirs += libs
            msg +=  ['* Using MKL library: %s' % library_dirs[-1]]
            #extra_link_args += ['-Wl,-rpath=' + library_dirs[-1]]
        else:
            atlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
                if glob(join(dir, 'libatlas.so')) != []:
                    atlas = True
                    libdir = dir        
                    break
            openblas = False
            for dir in ['/usr/lib', '/usr/local/lib']:
                if glob(join(dir, 'libopenblas.so')) != []:
                    openblas = True
                    libdir = dir        
                    break
            if openblas:  # prefer openblas
                libraries += ['openblas', 'lapack']
                library_dirs += [libdir]
                msg +=  ['* Using OpenBLAS library']
            else:
                if atlas:  # then atlas
                    # http://math-atlas.sourceforge.net/errata.html#LINK
                    # atlas does not respect OMP_NUM_THREADS - build single-thread
                    # http://math-atlas.sourceforge.net/faq.html#tsafe
                    libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                    library_dirs += [libdir]
                    msg +=  ['* Using ATLAS library']
                else:
                    libraries += ['blas', 'lapack']
                    msg +=  ['* Using standard lapack']

            # add libg2c if available
            g2c=False
            for dir in ['/usr/lib', '/usr/local/lib']:
                if glob(join(dir, 'libg2c.so')) != []:
                    g2c=True
                    break
                if glob(join(dir, 'libg2c.a')) != []:
                    g2c=True
                    break
            if g2c: libraries += ['g2c']

    elif sys.platform == 'darwin':

        extra_compile_args += ['-Wall', '-std=c99']
        include_dirs += ['/usr/include/malloc']

        if glob('/System/Library/Frameworks/vecLib.framework') != []:
            extra_link_args += ['-framework vecLib']
            msg += ['* Using vecLib']
        else:
            libraries += ['blas', 'lapack']
            msg +=  ['* Using standard lapack']

    # https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2012-May/001473.html
    p = platform.dist()
    if p[0].lower() in ['redhat', 'centos'] and p[1].startswith('6.'):
        define_macros.append(('_GNU_SOURCE', '1'))

    return msg


def get_parallel_config(mpi_libraries,mpi_library_dirs,mpi_include_dirs,
                        mpi_runtime_library_dirs,mpi_define_macros):

    globals = {}
    execfile('gpaw/mpi/config.py', globals)
    mpi = globals['get_mpi_implementation']()

    if mpi == '':
        mpicompiler = None

    elif mpi == 'sun':
        mpi_include_dirs += ['/opt/SUNWhpc/include']
        mpi_libraries += ['mpi']
        mpi_library_dirs += ['/opt/SUNWhpc/lib']
        mpi_runtime_library_dirs += ['/opt/SUNWhpc/lib']
        mpicompiler = get_config_var('CC')

    elif mpi == 'poe':
        mpicompiler = 'mpcc_r'

    else:
        #Try to use mpicc
        mpicompiler = 'mpicc'

    return mpicompiler

def get_scalapack_config(define_macros):
    # check ScaLapack settings
    define_macros.append(('GPAW_WITH_SL', '1'))

def get_hdf5_config(define_macros):
    # check ScaLapack settings
    define_macros.append(('GPAW_WITH_HDF5', '1'))


def mtime(path, name, mtimes):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned.
    This function fails if two include files with the same name
    are present in different directories."""

    include = re.compile('^#\s*include "(\S+)"', re.MULTILINE)

    if mtimes.has_key(name):
        return mtimes[name]
    t = os.stat(os.path.join(path, name))[ST_MTIME]
    for name2 in include.findall(open(os.path.join(path, name)).read()):
        path2, name22 = os.path.split(name2)
        if name22 != name:
            t = max(t, mtime(os.path.join(path, path2), name22, mtimes))
    mtimes[name] = t
    return t

def check_dependencies(sources):
    # Distutils does not do deep dependencies correctly.  We take care of
    # that here so that "python setup.py build_ext" always does the right
    # thing!
    mtimes = {}  # modification times

    # Remove object files if any dependencies have changed:
    plat = distutils.util.get_platform() + '-' + sys.version[0:3]
    remove = False
    for source in sources:
        path, name = os.path.split(source)
        t = mtime(path + '/', name, mtimes)
        o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
        if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
            print 'removing', o
            os.remove(o)
            remove = True

    so = 'build/lib.%s/_gpaw.so' % plat
    if os.path.exists(so) and remove:
        # Remove shared object C-extension:
        # print 'removing', so
        os.remove(so)

def test_configuration():
    raise NotImplementedError


def write_configuration(define_macros, include_dirs, libraries, library_dirs,
                        extra_link_args, extra_compile_args,
                        runtime_library_dirs, extra_objects, mpicompiler,
                    mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                    mpi_runtime_library_dirs, mpi_define_macros):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError, x:
        print x
        return
    print >> out, "Current configuration"
    print >> out, "libraries", libraries
    print >> out, "library_dirs", library_dirs
    print >> out, "include_dirs", include_dirs
    print >> out, "define_macros", define_macros
    print >> out, "extra_link_args", extra_link_args
    print >> out, "extra_compile_args", extra_compile_args
    print >> out, "runtime_library_dirs", runtime_library_dirs
    print >> out, "extra_objects", extra_objects
    if mpicompiler is not None:
        print >> out
        print >> out, "Parallel configuration"
        print >> out,  "mpicompiler", mpicompiler
        print >> out,  "mpi_libraries", mpi_libraries
        print >> out, "mpi_library_dirs", mpi_library_dirs
        print >> out, "mpi_include_dirs", mpi_include_dirs
        print >> out, "mpi_define_macros", mpi_define_macros
        print >> out, "mpi_runtime_library_dirs", mpi_runtime_library_dirs
    out.close()


def build_interpreter(define_macros, include_dirs, libraries, library_dirs,
                      extra_link_args, extra_compile_args,
                      runtime_library_dirs, extra_objects,
                      mpicompiler, mpilinker, mpi_libraries, mpi_library_dirs,
                      mpi_include_dirs, mpi_runtime_library_dirs,
                      mpi_define_macros):

    #Build custom interpreter which is used for parallel calculations

    cfgDict = get_config_vars()
    plat = distutils.util.get_platform() + '-' + sys.version[0:3]

    cfiles = glob('c/[a-zA-Z_]*.c') + ['c/bmgs/bmgs.c']
    cfiles += glob('c/xc/*.c')

    sources = ['c/bc.c', 'c/localized_functions.c', 'c/mpi.c', 'c/_gpaw.c',
               'c/operators.c', 'c/transformers.c',
               'c/blacs.c', 'c/utilities.c', 'c/hdf5.c']
    objects = ' '.join(['build/temp.%s/' % plat + x[:-1] + 'o'
                        for x in cfiles])

    if not os.path.isdir('build/bin.%s/' % plat):
        os.makedirs('build/bin.%s/' % plat)
    exefile = 'build/bin.%s/' % plat + '/gpaw-python'

    libraries += mpi_libraries
    library_dirs += mpi_library_dirs
    define_macros += mpi_define_macros
    include_dirs += mpi_include_dirs
    runtime_library_dirs += mpi_runtime_library_dirs

    define_macros.append(('PARALLEL', '1'))
    define_macros.append(('GPAW_INTERPRETER', '1'))
    macros = ' '.join(['-D%s=%s' % x for x in define_macros if x[0].strip()])

    include_dirs.append(cfgDict['INCLUDEPY'])
    include_dirs.append(cfgDict['CONFINCLUDEPY'])
    includes = ' '.join(['-I' + incdir for incdir in include_dirs])

    library_dirs.append(cfgDict['LIBPL'])
    lib_dirs = ' '.join(['-L' + lib for lib in library_dirs])

    libs = ' '.join(['-l' + lib for lib in libraries if lib.strip()])
    # See if there is "scalable" libpython available
    libpl = cfgDict['LIBPL']
    if glob(libpl + '/libpython*mpi*'):
        libs += ' -lpython%s_mpi' % cfgDict['VERSION']
    else:
        libs += ' -lpython%s' % cfgDict['VERSION']
    libs = ' '.join([libs, cfgDict['LIBS'], cfgDict['LIBM']])

    #Hack taken from distutils to determine option for runtime_libary_dirs
    if sys.platform[:6] == 'darwin':
        # MacOSX's linker doesn't understand the -R flag at all
        runtime_lib_option = '-L'
    elif sys.platform[:5] == 'hp-ux':
        runtime_lib_option = '+s -L'
    elif os.popen('mpicc --showme 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    elif os.popen('mpicc -show 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    else:
        runtime_lib_option = '-R'

    runtime_libs = ' '.join([ runtime_lib_option + lib for lib in runtime_library_dirs])

    extra_link_args.append(cfgDict['LDFLAGS'])
    if sys.platform in ['aix5', 'aix6']:
        extra_link_args.append(cfgDict['LINKFORSHARED'].replace('Modules', cfgDict['LIBPL']))
    elif sys.platform == 'darwin':
        pass
    else:
        extra_link_args.append(cfgDict['LINKFORSHARED'])

    # Compile the parallel sources
    for src in sources:
        obj = 'build/temp.%s/' % plat + src[:-1] + 'o'
        cmd = ('%s %s %s %s -o %s -c %s ' ) % \
              (mpicompiler,
               macros,
               ' '.join(extra_compile_args),
               includes,
               obj,
               src)
        print cmd
        if '--dry-run' not in sys.argv:
            error=os.system(cmd)
            if error != 0:
                msg = ['* compiling FAILED!  Only serial version of code will work.']
                break

    # Link the custom interpreter
    cmd = ('%s -o %s %s %s %s %s %s %s' ) % \
          (mpilinker,
           exefile,
           objects,
           ' '.join(extra_objects),
           lib_dirs,
           libs,
           runtime_libs,
           ' '.join(extra_link_args))

    msg = ['* Building a custom interpreter']
    print cmd
    if '--dry-run' not in sys.argv:
        error=os.system(cmd)
        if error != 0:
            msg += ['* linking FAILED!  Only serial version of code will work.']


    return error, msg
