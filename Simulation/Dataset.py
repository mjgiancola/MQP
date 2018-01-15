import numpy as np

from util.sinkhorn import *
from util.softmax import *

class Dataset():
  def __init__(self, numLabels, numLabelers, numImages, numCharacters,
               gamma, alphabet, priorZ, labels, probZ, Labelers, isDSM):
    self.numLabels = numLabels
    self.numLabelers = numLabelers
    self.numImages = numImages
    self.numCharacters = numCharacters
    self.gamma = gamma
    self.alphabet = alphabet
    self.priorZ = priorZ
    self.labels = labels
    self.probZ = probZ
    self.Labelers = Labelers
    self.isDSM = isDSM

  # For each labeler, compute style matrix S given parameterizing matrix A
  def computeStyle(self):
    if self.isDSM: self.computeStyle_DSM()
    else:          self.computeStyle_RSM()

  def computeStyle_DSM(self):
    for i in range(self.numLabelers):
      A = self.Labelers[i].A # Current parameterizing matrix
      self.Labelers[i].iterations = sink_norm( np.exp(A) )

      I = self.Labelers[i].iterations
      self.Labelers[i].style = I[len(I)-1] # Set style matrix

  def computeStyle_RSM(self):
    for i in range(self.numLabelers):
      A = self.Labelers[i].A
      self.Labelers[i].style = softmax(A)

  # For a large alphabet this will print a lot - consider piping to a file
  def outputResults(self):
    
    self.computeStyle()

    for i in range(self.numLabelers): 
      print "Style[%d]:" % i
      print self.Labelers[i].style
      print ""

    for y in range(self.numCharacters): 
      x = 0
      for z in self.probZ[y]:
        #print "P(Z(%d)=%d) = %f" % (x, y, z)
        print "P(Z(%d)=%c) = %f" % (x, chr(y + 97), z) # Should convert back to label
        x += 1
      print ""
