import numpy as np

from util.sinkhorn import *

EPSILON = 0.01

class Labeler:

  # A is the parameterizing matrix of S (style)
  # Style is A row normalized
  def __init__(self, A):
    self.A = A
    B = my_relu(A)
    C = row_norm(B)
    self.iterations = [A, B, C]
    self.style = C

# For each labeler, compute style matrix S given parameterizing matrix A
def computeStyle(data):
  for i in range(data.numLabelers):
    A = data.Labelers[i].A
    data.Labelers[i].style = row_norm(my_relu(A))

# Modified ReLU which forces strict positivity
def my_relu(m):
  return np.maximum(m, EPSILON)