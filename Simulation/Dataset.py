import numpy as np
import itertools

from Dataset import *
from util.Label import *
from util.sinkhorn import *
from util.softmax import *

EPSILON = 1e-5

# DATA FILE FORMAT for DATA #

# numLabels numLabelers numImages numCharacters
# c1 c2 .. cZ (characters in alphabet)
# priorZ1 priorZ2 ... priorZ<numCharacters>
# Image_i Labeler_j Label_ij
# ...
# Image_i Ground_Truth_Label_i <optional>
# ...


# gamma: Regularizing constant for prior on style
# isDSM: True if running SinkProp, False if running Softmax
# hasGT: True if datafile contains ground truth labels
class Dataset():
  def __init__(self, filename, gamma, isDSM, hasGT):

    self.isDSM = isDSM
    if isDSM: from SinkProp.Labeler import Labeler
    else:     from Stochastic.Labeler import Labeler

    fp = open(filename, 'r')

    # Read metadata
    line = fp.readline().strip().split()

    self.numLabels = int(line[0])
    self.numLabelers = int(line[1])
    self.numImages = int(line[2])
    self.numCharacters = int(line[3]) # The number of characters in the alphabet
    self.gamma = gamma
    self.alphabet = fp.readline().strip().split()

    # Read Z priors
    line = fp.readline().strip().split()
    self.priorZ = np.empty((self.numCharacters, self.numImages))
    for x in range(self.numCharacters):
      self.priorZ[x][:] = line[x]

    # Read in labels
    self.labels = []
    line = fp.readline()
    while line != "" and line != "\n":
      line = line.strip().split()
      
      # Image ID, Labeler ID, Label
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      self.labels.append(lbl)

      line = fp.readline()

    self.probZ = np.empty((self.numCharacters, self.numImages))
    priorA = np.identity(self.numCharacters)
    self.Labelers = [ Labeler(priorA) for i in range(self.numLabelers) ]

    if hasGT:
      self.gt = []
      line = fp.readline()
      while line != "" and line != "\n":
        line = line.strip().split()
        self.gt.append(int(line[1])) # Only store label
        line = fp.readline()

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

  def percent_correct(self):
    # Compute given labels based on greatest probability
    given = np.argmax(self.probZ, axis=0)
    return self.acc(given)

  # Computes the highest possible percent correct by considering all possible
  # permutations of cluster names
  def best_percent_correct(self):

      # Compute given labels based on greatest probability
      given = np.argmax(self.probZ, axis=0)

      # Generate list of permutations of character set
      permutations = list(itertools.permutations(range(self.numCharacters)))
      
      acc = 0
      best_perm = 0

      for perm in permutations:
        labels = np.empty(given.shape)
        i = 0
        for lbl in given:
          labels[i] = perm[lbl]
          i += 1

        new_acc = self.acc(labels)
        if new_acc > acc:
          acc = new_acc
          best_perm = perm

      print "Transformation of labels: " + str(best_perm)
      return acc

  # Compute percent correct by comapring values in given array to gt (ground truth)
  def acc(self, given):
    correct = 0.
    total = 0.

    for i in range(self.numImages):
      if given[i] == self.gt[i]:
        correct += 1
      total += 1

    return correct / total

  def cross_entropy(self):
    y_hats = self.probZ
    y_hats[y_hats == 0] = EPSILON # Model has ability to achieve probZ 0 and 1
    y_hats[y_hats == 1] = 1 - EPSILON
    y_actuals = self.gt_to_onehot()

    m = self.numImages
    ones = np.ones(y_actuals.shape)

    positives = y_actuals * np.log(y_hats)
    negatives = (ones - y_actuals) * np.log(ones - y_hats)

    return -1./m * np.sum(positives + negatives)

  def gt_to_onehot(self):
    onehot = np.zeros((self.numImages, self.numCharacters))
    onehot[np.arange(self.numImages), self.gt] = 1
    return np.transpose(onehot)

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
      