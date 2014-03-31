#!/usr/bin/env python

# Generate a MPICH_RANK_ORDER file for Cray systems.
# User has to then set the environment variable 
# MPICH_RANK_REORDER_METHOD to the value 3,
# (e.g.  export  MPICH_RANK_REORDER_METHOD=3)
# so that MPI ranks are placed to physical
# CPU cores according to MPICH_RANK_ORDER.
#
# Rank reordering should be used only in the case of ground state
# calculations with band parallelization, where ranks will be ordered
# so that ranks belonging to the same domain but different band group
# are placed physically close to each other.
#
# Rank reordering can increase the performance by 20-30 % in some cases.
#
# Usage: mapfile_cray.py total_cpus band_cpus
#
# e.g. mapfile_cray.py 1024 2

from sys import argv
import os
import numpy as np
cores = int(argv[1])
band_cores = int(argv[2])
ndomains = cores // band_cores
assert ndomains*band_cores == cores
print "# Total number of cores:", cores
print "# Cores for band parallelization:", band_cores
print "# Cores for domain decomposition:", ndomains

reorder_file = open('MPICH_RANK_ORDER', 'w')

for rd in range(ndomains):
    band_ranks = range(rd, rd + (ndomains * band_cores), ndomains)
    rank_string = ""
    for rb in band_ranks[:-1]:
       rank_string += str(rb) + ","
    rank_string += str(band_ranks[-1])
    print >> reorder_file, rank_string

reorder_file.close()
