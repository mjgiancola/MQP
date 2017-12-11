from scipy.optimize import check_grad
from time import time
import sys

from EM import *
from Dataset import *

# Verifies the correctness of Q and gradQ
def check_gradient(data):
  EStep(data)

  x0 = packX(data)
  print check_grad(f, df, x0, data)

if __name__=='__main__':
  if len(sys.argv) < 2:
    print "Usage: python simulation.py <data>"
    print "where <data> is the filename of a data file which is formatted as described in the README file."
    exit()

  data = Dataset(sys.argv[1])
  #check_gradient(data)

  start = time()
  EM(data)
  elapsed = time() - start
  data.outputResults()
  print "Completed in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)