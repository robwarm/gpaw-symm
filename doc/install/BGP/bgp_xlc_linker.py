#!/usr/bin/env python
"""bgp_xlc.py is a wrapper for the BGP xlc compiler,
   converting/removing incompatible gcc args.   """

import sys
from subprocess import call
from glob import glob

args2change = {"-fno-strict-aliasing":"",
               "-fmessage-length=0":"",
               "-Wall":"",
               "-std=c99":"-qlanglvl=extc99",
               "-fPIC":"",
               "-g":"",
               "-D_FORTIFY_SOURCE=2":"",
               "-DNDEBUG":"",
               "-UNDEBUG":"",
               "-pthread":"",
               "-shared":"-qmkshrobj",
               "-Xlinker":"",
               "-export-dynamic":"",
               "-Wstrict-prototypes":"",
               "-dynamic":"",
               "-O3":"",
               "-O2":"",
               "-O1":"",
               "-fwrapv":""}

fragile_files = ["test.c"]
qhot_files = ["c/blas.c", "c/utilities.c","c/lfc.c","c/localized_functions.c"]
non_c99files = glob('c/libxc/src/*.c')

cmd = ""
opt = 1

for arg in sys.argv[1:]:
    cmd += " "
    t = arg.strip()
    if t in fragile_files:
        opt = 2
    if t in non_c99files:
        opt = 3
    if t in qhot_files:
        opt = 4
    if t in args2change:
        cmd += args2change[t]
    else:
        cmd += arg

flags_list = {1: "-g -O3 -qlanglvl=extc99 -qflag=w:w -qpic",
              2: "-g -O3 -qstrict -qlanglvl=extc99 -qflag=w:w -qpic",
              3: "-g -O3 -qflag=w:w -qpic",
              4: "-g -O3 -qhot -qlanglvl=extc99 -qflag=w:w -qpic",
              }

flags = flags_list[opt]
# make sure this matches the version of the XL compiler in bgp_xlc.py
# as well as the libraries in customize_surveyor_xlc.py
cmd = "/soft/apps/ibmcmp-aug2011/vac/bg/9.0/bin/bgxlc_r %s %s"%(flags, cmd)

print "\nexecmd: %s\n"%cmd
call(cmd, shell=True)
