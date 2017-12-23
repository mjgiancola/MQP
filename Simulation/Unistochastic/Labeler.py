import numpy as np

from util.sinkhorn import *
from Dataset import *

class Labeler:

  # A is the parameterizing matrix of S (style)
  # Style is A row normalized
  def __init__(self, A):
    self.A = A
    self.style = softmax(A)
