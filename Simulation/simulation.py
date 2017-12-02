import sys

from EM import *
from Dataset import *

if __name__=='__main__':
  if len(sys.argv) < 2:
    print "Usage: python simulation.py <data>"
    print "where <data> is the filename of a data file which is formatted as described in the README file."
    exit()

  data = Dataset(sys.argv[1])
  EM(data)
  data.outputResults()
  