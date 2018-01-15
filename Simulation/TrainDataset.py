import numpy as np

from Dataset import *
from util.Label import *

# DATA FILE FORMAT for TRAINING DATA #

# numLabels numLabelers numImages numCharacters
# c1 c2 .. cZ (characters in alphabet)
# priorZ1 priorZ2 ... priorZ<numCharacters>
# Image_i Labeler_j Label_ij
# ...

class TrainDataset(Dataset):
  def __init__(self, filename, gamma, isDSM):

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
    while line != "":
      line = line.strip().split()
      
      # Image ID, Labeler ID, Label
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      labels.append(lbl)

      line = fp.readline()

    probZ = np.empty((numCharacters, numImages))
    priorA = np.identity(numCharacters)
    Labelers = [ Labeler(priorA) for i in range(numLabelers) ]

    Dataset.__init__(self, numLabels, numLabelers, numImages, numCharacters,
                     gamma, alphabet, priorZ, labels, probZ, Labelers, isDSM)
