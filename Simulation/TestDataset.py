import numpy as np

from Dataset import *
from util.Label import *

# DATA FILE FORMAT for TESTING DATA#

# numLabels numLabelers numImages
# Image_i Labeler_j Label_ij
# ...
# Image_i Ground_Truth_Label_i
# ...

class TestDataset(Dataset):
  def __init__(self, filename, train_data, isDSM):

    if isDSM: from SinkProp.Labeler import Labeler
    else:     from Stochastic.Labeler import Labeler

    fp = open(filename, 'r')

    # Read metadata
    line = fp.readline().strip().split()

    numLabels = int(line[0])
    numLabelers = int(line[1])
    numImages = int(line[2])

    # These are fixed by the corresponding training data
    numCharacters = train_data.numCharacters
    gamma = train_data.gamma
    alphabet = train_data.alphabet
    Labelers = train_data.Labelers
    priorZ = np.empty((numCharacters, numImages))
    for x in range(numCharacters):
      priorZ[x][:] = train_data.priorLine[x]

    # Read in labels
    labels = []
    line = fp.readline()
    while line != "" and line != "\n":
      line = line.strip().split()
      
      # Image ID, Labeler ID, Label
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      labels.append(lbl)

      line = fp.readline()

    # Initialize empty aaaaaa
    probZ = np.empty((numCharacters, numImages))

    Dataset.__init__(self, numLabels, numLabelers, numImages, numCharacters,
                     gamma, alphabet, priorZ, labels, probZ, Labelers, isDSM)

    self.gt = []
    line = fp.readline()
    while line != "" and line != "\n":
      line = line.strip().split()
      self.gt.append(int(line[1])) # Only store label
      line = fp.readline()

  def percent_correct(self):

    # Compute given labels based on greatest probability
    given = np.argmax(self.probZ, axis=0)

    correct = 0.
    total = 0.

    for i in range(self.numImages):
      if given[i] == self.gt[i]:
        correct += 1
      total += 1

    return correct / total