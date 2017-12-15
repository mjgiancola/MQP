import numpy as np

from util.sinkhorn import *

class Labeler:

  # A is the parameterizing matrix of S (style)
  # iterations is a list of numpy matrices
  # where iterations[0] = exp(A)
  #       iterations[1] = row_norm(exp(A))
  #       iterations[2] = col_norm(row_norm(exp(A)))
  #       ...
  #       iterations[n-1] = DSM
  # as constructed by sink_norm
  def __init__(self, A):
    self.A = A
    self.iterations = sink_norm(np.exp(A)) # Comment out for no SP
    #self.style = self.iterations[len(self.iterations)-1] # Comment out for no SP
    #self.style = A + 0.01

# For each labeler, compute style matrix S given parameterizing matrix A
def computeStyle(data):
  for i in range(data.numLabelers):

    A = data.Labelers[i].A # Current parameterizing matrix
    data.Labelers[i].iterations = sink_norm( np.exp(A) ) # Comment out for no SP

    # Comment out for no SP
    I = data.Labelers[i].iterations
    data.Labelers[i].style = I[len(I)-1] # Set style matrix

    # Only for running without SP
    #data.Labelers[i].style = A + 0.01
