import sys
from Dataset import *

# Returns the most-likely style matrix given the labels
def optimizeStyle(labelerID, labels, truth):
  pass

if __name__=='__main__':
  if len(sys.argv) < 2:
    print "Usage: python style.py <data>"
    print "where <data> is the filename of a data file which is formatted as described in the README file."
    exit()

  labels = []
  labels.append(Label(0, 0, "abc"))
  labels.append(Label(1, 0, "aaa"))
  labels.append(Label(2, 0, "acc"))
  labels.append(Label(3, 0, "abc"))
  labels.append(Label(4, 0, "ccc"))
  
  truth = []
  truth.append("aaa")
  truth.append("aba")
  truth.append("baa")
  truth.append("acc")
  truth.append("cca")
  