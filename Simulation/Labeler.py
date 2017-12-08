import numpy as np

from sinkhorn import *

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
    self.iterations = sink_norm(np.exp(A))
    self.style = self.iterations[len(self.iterations)-1]

# For each labeler, compute style matrix S given parameterizing matrix A
def computeStyle(data):
  for i in range(data.numLabelers):

    A = data.Labelers[i].A # Current parameterizing matrix
    data.Labelers[i].iterations = sink_norm( np.exp(A) )

    I = data.Labelers[i].iterations
    data.Labelers[i].style = I[len(I)-1] # Set style matrix