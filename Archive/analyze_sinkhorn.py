import numpy as np
from numpy.random import rand

from util.sinkhorn import *

def compute_error(DSM):
  row_sums = np.sum(DSM, axis=1)
  col_sums = np.sum(DSM, axis=0)

  return np.sum( (np.abs(row_sums - 1) + np.abs(col_sums - 1)) )

if __name__=='__main__':
  # parser = argparse.ArgumentParser(description='Test Sinkhorn Normalization with varying input.')
  # parser.add_argument('c', help='Number of clusters (ie. dimensions of matrix)')
  # args = parser.parse_args()

  for c in range(3,7):
    A = rand(c, c)
    errors = []

    for sink_iter in range(1, 11):
      iterations = sink_norm(A, sink_iter)
      DSM = iterations[len(iterations)-1]
      errors.append(compute_error(DSM))

    print "Error for %d by %d DSM:" % (c,c)
    for i in range(1, len(errors)):
      print "n=%d: %f" % (i, errors[i])
