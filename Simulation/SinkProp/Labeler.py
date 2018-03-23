import numpy as np

from Normalization.sinkhorn import *

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
    self.style = self.iterations[len(self.iterations)-1]
    
