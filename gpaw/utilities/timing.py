# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""A replacement for the ``time.clock()`` function.

From the clock man page::

       Note that the time can wrap around. On a 32bit system
       where CLOCKS_PER_SEC equals 1000000 this function will
       return the same value approximately every 72 minutes.

The ``clock()`` function defined below tries to fix this problem.
However, if the ``clock()`` function is not called often enough (more
than 72 minutes between two calls), then there is no way of knowing
how many times the ``time.clock()`` function has wrapped arround! - in
this case a huge number is returned (1.0e100).  This problem can be
avoided by calling the ``update()`` function at intervals smaller than
72 minutes."""

import time
import math
import sys

import numpy as np

import gpaw.mpi as mpi
MASTER = 0

wrap = 1e-6 * 2**32

# Global variables:
c0 = time.clock()
t0 = time.time()
cputime = 0.0
trouble = False


def clock():
    """clock() -> floating point number

    Return the CPU time in seconds since the start of the process."""

    update()
    if trouble:
        return 1.0e100
    return cputime

def update():
    global trouble, t0, c0, cputime
    if trouble:
        return
    t = time.time()
    c = time.clock()
    if t - t0 >= wrap:
        trouble = True
        return
    dc = c - c0
    if dc < 0.0:
        dc += wrap
    cputime += dc
    t0 = t
    c0 = c

def function_timer(func, *args, **kwargs):
    out = kwargs.pop('timeout', sys.stdout)
    t1 = time.time()
    r = func(*args, **kwargs)
    t2 = time.time()
    print >>out, t2 - t1
    return r

    
class Timer:
    def __init__(self, print_levels=1000):
        self.timers = {}
        self.t0 = time.time()
        self.running = []
        self.print_levels = print_levels
    
    def print_info(self, calc):
        """Override to get to write info during calculator's initialize()."""
        pass

    def start(self, name):
        names = tuple(self.running + [name])
        self.timers[names] = self.timers.get(names, 0.0) - time.time()
        self.running.append(name)
        
    def stop(self, name=None):
        if name is None: name = self.running[-1]
        names = tuple(self.running)
        running = self.running.pop()
        if name != running:
            raise RuntimeError('Must stop timers by stack order.  '
                               'Requested stopping of %s but topmost is %s'
                               % (name, running))
        self.timers[names] += time.time()
            
    def get_time(self, *names):
#        print self.timers, names
        return self.timers[names]
                
    def write(self, out=sys.stdout):
        while self.running:
            self.stop()
        if len(self.timers) == 0:
            return

        t0 = time.time()
        tot = t0 - self.t0

        n = max([len(names[-1]) + len(names) for names in self.timers]) + 1
        line = '=' * (n + 26) + '\n'
        out.write(line)
        out.write('%-*s    incl.     excl.\n' % (n, 'Timing:'))
        out.write(line)
        tother = tot
        
        inclusive = self.timers.copy()

        exclusive = self.timers
        keys = exclusive.keys()
        keys.sort()
        for names in keys:
            t = exclusive[names]
            if len(names) > 1:
                if len(names) < self.print_levels + 1:
                    exclusive[names[:-1]] -= t
            else:
                tother -= t
        exclusive[('Other',)] = tother
        inclusive[('Other',)] = tother
        keys.append(('Other',))
        for names in keys:
            t = exclusive[names]
            tinclusive = inclusive[names]
            r = t / tot
            p = 100 * r
            i = int(40 * r + 0.5)
            if i == 0:
                bar = '|'
            else:
                bar = '|%s|' % ('-' * (i - 1))
            level = len(names)
            if level > self.print_levels:
                continue
            name = (level - 1) * ' ' + names[-1] + ':'
            out.write('%-*s%9.3f %9.3f %5.1f%% %s\n' %
                      (n, name, tinclusive, t, p, bar))
        out.write(line)
        out.write('%-*s%9.3f %5.1f%%\n' % (n + 10, 'Total:', tot, 100.0))
        out.write(line)
        out.write('date: %s\n' % time.asctime())

    def add(self, timer):
        for name, t in timer.timers.items():
            self.timers[name] = self.timers.get(name, 0.0) + t


class NullTimer:
    """Compatible with Timer and StepTimer interfaces.  Does nothing."""
    def __init__(self): pass
    def print_info(self, calc): pass
    def start(self, name): pass
    def stop(self, name=None): pass
    def get_time(self, name):
        return 0.0
    def write(self, out=sys.stdout): pass
    def write_now(self, mark=''): pass
    def add(self, timer): pass


nulltimer = NullTimer()


class DebugTimer(Timer):
    def __init__(self, print_levels=1000, comm=mpi.world, txt=sys.stdout):
        Timer.__init__(self, print_levels)
        ndigits = 1 + int(math.log10(comm.size))
        self.srank = '%0*d' % (ndigits, comm.rank)
        self.txt = txt

    def start(self, name):
        Timer.start(self, name)
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s >> %15.8f %s (%7.5fs) started\n'
                       % (self.srank, abstime, name, t))

    def stop(self, name=None):
        if name is None:
            name = self.running[-1]
        abstime = time.time()
        t = self.timers[tuple(self.running)] + abstime
        self.txt.write('T%s << %15.8f %s (%7.5fs) stopped\n'
                       % (self.srank, abstime, name, t))
        Timer.stop(self, name)


class ParallelTimer(DebugTimer):
    """Like DebugTimer but writes timings from all ranks.

    Each rank writes to timings.<rank>.txt.  Also timings.metadata.txt
    will contain information about the parallelization layout.  The idea
    is that the output from this timer can be used for plots and to
    determine bottlenecks in the parallelization.

    See the tool gpaw-plot-parallel-timings."""
    def __init__(self):
        ndigits = len(str(mpi.world.size - 1))
        ranktxt = '%0*d' % (ndigits, mpi.world.rank)
        fname = 'timings.%s.txt' % ranktxt
        txt = open(fname, 'w')
        DebugTimer.__init__(self, comm=mpi.world, txt=txt)

    def print_info(self, calc):
        """Print information about parallelization into a file."""
        fd = open('timings.metadata.txt', 'w')
        DebugTimer.print_info(self, calc)
        wfs = calc.wfs
        
        # We won't have to type a lot if everyone just sends all their numbers.
        myranks = np.array([wfs.world.rank, wfs.kd.comm.rank,
                            wfs.bd.comm.rank, wfs.gd.comm.rank])
        allranks = None
        if wfs.world.rank == 0:
            allranks = np.empty(wfs.world.size * 4, dtype=int)
        wfs.world.gather(myranks, 0, allranks)
        if wfs.world.rank == 0:
            for itsranks in allranks.reshape(-1, 4):
                fd.write('r=%d k=%d b=%d d=%d\n' % tuple(itsranks))
        fd.close()


class StepTimer(Timer):
    """Step timer to print out timing used in computation steps.
    
    Use it like this::

      from gpaw.utilities.timing import StepTimer
      st = StepTimer()
      ...
      st.write_now('step 1')
      ...
      st.write_now('step 2')

    The parameter write_as_master_only can be used to force the timer to
    print from processess that are not the mpi master process.
    """
    
    def __init__(self, out=sys.stdout, name=None, write_as_master_only=True):
        Timer.__init__(self)
        if name is None:
            name = '<%s>' % sys._getframe(1).f_code.co_name
        self.name = name
        self.out = out
        self.alwaysprint = not write_as_master_only
        self.now = 'temporary now'
        self.start(self.now)


    def write_now(self, mark=''):
        self.stop(self.now)
        if self.alwaysprint or mpi.rank == MASTER:
            print >> self.out, self.name, mark, self.get_time(self.now)
        self.out.flush()
        del self.timers[self.now]
        self.start(self.now)


class TAUTimer(Timer):
    """TAUTimer requires installation of the TAU Performance System
    http://www.cs.uoregon.edu/research/tau/home.php

    The TAU Python API will not output any data if there are any
    unmatched starts/stops in the code."""

    top_level = 'GPAW.calculator' # TAU needs top level timer 
    merge = True # Requires TAU 2.19.2 or later

    def __init__(self):
        Timer.__init__(self)
        import pytau
        self.pytau = pytau
        self.tau_timers = {}
        pytau.setNode(mpi.rank)
        self.tau_timers[self.top_level] = pytau.profileTimer(self.top_level)
        pytau.start(self.tau_timers[self.top_level])

    def start(self, name):
        Timer.start(self, name)
        self.tau_timers[name] = self.pytau.profileTimer(name)
        self.pytau.start(self.tau_timers[name])
        
    def stop(self, name=None):
        Timer.stop(self, name)
        self.pytau.stop(self.tau_timers[name])

    def write(self, out=sys.stdout):
        Timer.write(self, out)
        if self.merge:
            self.pytau.dbMergeDump()
        else:
            self.pytau.stop(self.tau_timers[self.top_level])

        
class HPMTimer(Timer):
    """HPMTimer requires installation of the IBM BlueGene/P HPM
    middleware interface to the low-level UPC library. This will
    most likely only work at ANL's BlueGene/P. Must compile
    with GPAW_HPM macro in customize.py. Note that HPM_Init
    and HPM_Finalize are called in _gpaw.c and not in the Python
    interface. Timer must be called on all ranks in node, otherwise
    HPM will hang. Hence, we only call HPM_start/stop on a list
    subset of timers."""
    
    top_level = 'GPAW.calculator' # HPM needs top level timer
    compatible = ['Initialization','SCF-cycle'] 

    def __init__(self):
        Timer.__init__(self)
        from _gpaw import hpm_start, hpm_stop
        self.hpm_start = hpm_start
        self.hpm_stop = hpm_stop
        hpm_start(self.top_level)

    def start(self, name):
        Timer.start(self, name)
        if name in self.compatible:
            self.hpm_start(name)
        
    def stop(self, name=None):
        Timer.stop(self, name)
        if name in self.compatible:
            self.hpm_stop(name)

    def write(self, out=sys.stdout):
        Timer.write(self, out)
        self.hpm_stop(self.top_level)

class CrayPAT_timer(Timer):
    """Interface to CrayPAT API. In addition to regular timers,
    the corresponding regions are profiled by CrayPAT. The gpaw-python has
    to be compiled under CrayPAT.
    """

    def __init__(self, print_levels=4):
        Timer.__init__(self, print_levels)
        from _gpaw import craypat_region_begin, craypat_region_end
        self.craypat_region_begin = craypat_region_begin
        self.craypat_region_end = craypat_region_end
        self.regions = {}
        self.region_id = 5 # leave room for regions in C

    def start(self, name):
        Timer.start(self, name)
        if self.regions.has_key(name):
            id = self.regions[name]
        else:
            id = self.region_id
            self.regions[name] = id
            self.region_id += 1
        self.craypat_region_begin(id, name)

    def stop(self, name=None):
        Timer.stop(self, name)
        id = self.regions[name]
        self.craypat_region_end(id)

