import sys
from numpy import test

from gpaw.mpi import rank

_stdout = sys.stdout
_stderr = sys.stderr
# numpy tests write to stderr
sys.stderr = open("numpy_test%02d.out" % rank, "w")
result = test(verbose=10)
sys.stdout = _stdout
sys.stderr = _stderr
if not result.wasSuccessful():
    print >> sys.stderr, "numpy_test%02d.out" % rank, result.errors, result.failures
assert result.wasSuccessful()

