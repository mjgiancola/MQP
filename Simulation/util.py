import numpy as np
from numpy.random import randint

NUM_LETTERS = 3 # Size of alphabet

# Given an alphabetic LOWER-CASE character c, returns
# the equivalent one-hot column vector
def charToOneHot(c):
  idx = ord(c) - 97
  one_hot = np.append(np.zeros(idx), np.ones(1))
  one_hot = np.append(one_hot, np.zeros( (NUM_LETTERS-1) - idx))
  assert(len(one_hot) == NUM_LETTERS)
  return one_hot.reshape((NUM_LETTERS,1))
  
# Given a one-hot vector, returns the corresponding
# lower case character
def oneHotToChar(one_hot):
  return chr(np.nonzero(one_hot)[0][0] + 97)

# Given weights, an array of weights (that don't necessarily sum to 1)
# Returns an index into the array, randomly selected by weights
def getWeightedRand(weights):
  weight_sum = 0
  for w in weights:
    weight_sum += w

  rnd = randint(weight_sum-1) + 1 # Random int between [1, weight_sum] (don't want zero)

  for i in range(len(weights)):
    if rnd < weights[i]:
      return i
    rnd -= weights[i]

  assert 1 != 1 # Shouldn't get here

# Pack style matrices into a vector for CGS
def packX(data):
  x = np.empty(0)
  for i in range(data.numLabelers):
    x = np.append(x, data.style[i].flatten())
  return x

# Unpack style matrices from x
def unpackX(x, data):
  for i in range(data.numLabelers):
    c = data.numCharacters
    mat_size = c ** 2
    style = np.matrix([x[y] for y in range(i*mat_size, (i*mat_size)+mat_size)])
    data.style[i] = style.reshape((c,c))
    