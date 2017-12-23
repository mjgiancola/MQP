import numpy as np

from util.Label import *
from util.softmax import *
from Labeler import *

EPSILON = 0.01

# DATA FILE FORMAT #

# numLabels numLabelers numImages numCharacters
# c1 c2 .. cZ (characters in alphabet)
# priorZ1 priorZ2 ... priorZ<numCharacters>
# Image_i Labeler_j Label_ij
# ...

class Dataset():
  def __init__(self, filename):

    fp = open(filename, 'r')

    # Read metadata
    line = fp.readline().strip().split()

    # TODO MIKE DO NOT FORGET THAT YOU CHANGED THE METADATA
    self.numLabels = int(line[0])
    self.numLabelers = int(line[1])
    self.numImages = int(line[2])
    self.numCharacters = int(line[3]) # The number of characters in the alphabet
    self.gamma = 0

    priorA = np.identity(self.numCharacters) + EPSILON
    self.Labelers = [ Labeler(priorA) for i in range(self.numLabelers) ]

    # Read alphabet
    self.alphabet = fp.readline().strip().split()

    # Read Z priors
    line = fp.readline().strip().split()

    self.priorZ = []
    for x in range(self.numCharacters):
      # Assuming prior is the same for all images
      self.priorZ.append([ float(line[x]) for y in range(self.numImages) ])

    self.labels = []
    # Read in labels
    line = fp.readline()
    while line != "":
      line = line.strip().split()
      
      # Image ID, Labeler ID, Label
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      self.labels.append(lbl)

      line = fp.readline()

    # Empty lists to store these later
    self.probZ = []
    for x in range(self.numCharacters):
      self.probZ.append([])

  # For each labeler, compute style matrix S given parameterizing matrix A
  def computeStyle(self):
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
