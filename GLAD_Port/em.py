import sys, math

from Dataset import *
from prob_functions import *

### TODO ###
# Write EStep in prob_functions
# Write computeQ in prob_functions
# Write MStep in prob_functions

def EM(data):
  THRESHOLD = 1E-5

  # Initialize parameters to priors
  data.alpha = [data.priorAlpha[i] for i in range(data.numLabelers)]
  data.beta  = [data.priorBeta[i]  for i in range(data.numImages)]

  EStep(data) # Estimate P(Z|L,alpha,beta) based on priors
  Q = computeQ(data) # Q is the auxilary function which is maximized (minimized?) in MStep

  # Iterate the EM algorithm until delta_Q is under a predefined threshold
  while True:
    lastQ = Q

    EStep(data)        # Re-estimate P(Z|L,alpha,beta)
    Q = computeQ(data) # TODO- Is this necessary? Possibly... based on how MStep works in original code
    print "First Q:"
    print Q
    MStep(data)        # Maximize Q
    Q = computeQ(data)
    print "Second Q:"
    print Q

    # # TODO Remove and uncomment code below
    # print "we made it!"
    # break

    # Since Python doesn't have do-while loops..
    if (math.fabs((Q - lastQ)/lastQ) < THRESHOLD):
     break

if __name__=='__main__':
  if len(sys.argv) < 2:
    print "Usage: python em.py <data>"
    print "where <data> is the filename of a data file which is formatted as described in the README file."
    exit()

  data = Dataset(sys.argv[1])
  EM(data)
  #data.outputResults()