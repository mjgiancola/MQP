import numpy as np
import itertools

from Dataset import *
from util.Label import *
from util.sinkhorn import *
from util.softmax import *

EPSILON = 1e-5

# DATA FILE FORMAT #

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
  def __init__(self, numLabels, numLabelers, numImages, numCharacters, gamma,
               alphabet, priorZ, labels, probZ, Labelers, hasGT, gt, isDSM):
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
    self.hasGT = hasGT
    self.gt = gt
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

  # Computes the highest possible percent correct and cross-entropy 
  # by considering all permutations of cluster names
  # ce is a boolean which, if True, indicates to compute cross-entropy
  def permutedAcc(self, crossEntropy=True):
    # Compute observed labels based on greatest probability
    observed = np.argmax(self.probZ, axis=0)

    # Generate list of permutations of character set
    permutations = list(itertools.permutations(range(self.numCharacters)))
    
    acc = 0; ce = -1
    best_perm = 0

    for perm in permutations:
      
      # Compute accuracy
      labels = np.empty(observed.shape)
      i = 0
      for lbl in observed:
        labels[i] = perm[lbl]
        i += 1
      new_acc = self.percent_correct(labels)

      if crossEntropy:
        # Compute cross-entropy
        y_hats = self.probZ[np.array(perm)]
        y_actuals = self.gt_to_onehot()
        new_ce  = self.cross_entropy(y_hats, y_actuals)
        
      if new_acc > acc:
        acc = new_acc
        if crossEntropy: ce  = new_ce
        best_perm = perm

    print "Transformation of labels: " + str(best_perm)
    return acc, ce

  # Computed without permuting labels
  def std_percent_correct(self):
      # Compute observed labels based on greatest probability
      observed = np.argmax(self.probZ, axis=0)
      return self.percent_correct(observed)

  # Compute percent correct by comapring values in given array to gt (ground truth)
  def percent_correct(self, given):
    correct = 0.
    total = 0.

    for i in range(self.numImages):
      if given[i] == self.gt[i]:
        correct += 1
      total += 1

    return correct / total

  # Computed without permuting labels
  def std_cross_entropy(self):
    y_hats = self.probZ
    y_actuals = self.gt_to_onehot()
    return self.cross_entropy(y_hats, y_actuals)

  # Computes cross_entropy
  def cross_entropy(self, y_hats, y_actuals):
    y_hats[y_hats == 0] = EPSILON # Model has ability to achieve probZ 0 and 1
    y_hats[y_hats == 1] = 1 - EPSILON

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
      
  # Outputs just styles
  # Helpful when you have lots of examples/labels but small number of annotators / small alphabet
  def outputStyles(self):
    
    self.computeStyle()

    for i in range(self.numLabelers): 
      print "Style[%d]:" % i
      print self.Labelers[i].style
      print ""

def init_from_file(filename, gamma, isDSM, hasGT):

  if isDSM: from SinkProp.Labeler import Labeler
  else:     from Stochastic.Labeler import Labeler

  fp = open(filename, 'r')

  # Read metadata
  line = fp.readline().strip().split()

  numLabels = int(line[0])
  numLabelers = int(line[1])
  numImages = int(line[2])
  numCharacters = int(line[3]) # The number of characters in the alphabet
  gamma = gamma
  alphabet = fp.readline().strip().split()

  # Read Z priors
  line = fp.readline().strip().split()
  priorZ = np.empty((numCharacters, numImages))
  for x in range(numCharacters):
    priorZ[x][:] = line[x]

  # Read in labels
  labels = []
  line = fp.readline()
  while line != "" and line != "\n":
    line = line.strip().split()
    
    # Image ID, Labeler ID, Label
    lbl = Label(int(line[0]), int(line[1]), int(line[2]))
    labels.append(lbl)

    line = fp.readline()

  probZ = np.zeros((numCharacters, numImages))
  priorA = np.identity(numCharacters)
  Labelers = [ Labeler(priorA) for i in range(numLabelers) ]

  gt = []
  if hasGT:
    line = fp.readline()
    while line != "" and line != "\n":
      line = line.strip().split()
      gt.append(int(line[1])) # Only store label
      line = fp.readline()

  # Initialize Dataset object
  return Dataset(numLabels, numLabelers, numImages, numCharacters, gamma,
                 alphabet, priorZ, labels, probZ, Labelers, hasGT, gt, isDSM)

# This is old, not being used currenlty but kept around in case useful
# NOTE: prior here is a float (assumed to be same over all images/characters)
def init_for_trials(numLabels, numLabelers, numImages, numCharacters, gamma, alphabet, prior, labels, isDSM):

  if isDSM: from SinkProp.Labeler import Labeler
  else:     from Stochastic.Labeler import Labeler

  priorZ = np.ones((numCharacters, numImages)) * prior

  probZ = np.empty((numCharacters, numImages))
  priorA = np.identity(numCharacters)
  Labelers = [ Labeler(priorA) for i in range(numLabelers) ]

  # Labels and gt are initialized to empty lists (filled later)
  # Assumed we have ground truth labels if running trials
  return Dataset(numLabels, numLabelers, numImages, numCharacters, gamma,
                 alphabet, priorZ, labels, probZ, Labelers, True, [], isDSM)
