import numpy as np
from numpy.random import randint
from scipy.stats import mode

def naive(data):

  # -1 will represent a missing label in this function
  labels = - np.ones((data.numLabelers, data.numImages))
  permutations = np.empty((data.numLabelers, data.numCharacters))

  # Populate labels matrix
  for idx in range(data.numLabels):
      i = data.labels[idx].labelerId
      j = data.labels[idx].imageIdx
      lij = data.labels[idx].label
      labels[i][j] = lij

  # Index of annotator who everyone else's permutations will be based on
  leader = randint(data.numLabelers)
  leaderLbls = labels[leader]

  for k in range(data.numCharacters):
    indices = np.where(leaderLbls == k)

    # For each labelers (besdies the leader), find the symbol (perm) used most 
    # often to represent k (based on the leader's labeling)
    for j in range(data.numLabelers):
      if j == leader: continue
      perm, _ = mode(labels[j][indices])
      permutations[j][perm[0]] = k # perm[0] because mode returns a list of lists (in case of ties I think..)

  # Un-permute everyone's labels (besides the leader)
  for j in range(data.numLabelers):
    if j == leader: continue

    # Find indices where an annotator uses somes symbol k
    indices = []
    for k in range(data.numCharacters):
      indices.append(np.where(labels[j] == k))
  
    # Unpermute those labels based on the permutation computed above
    for k in range(data.numCharacters):
      idx = indices[k]
      labels[j][idx] = permutations[j][k]

  # Compute the majority vote based on the unpermuted labels
  majority_vote, _ = mode(labels)
  majority_vote = majority_vote[0] # mode returns a list of lists

  # Compute percent correct
  correct = 0.
  total = 0.

  for i in range(data.numImages):
    if majority_vote[i] == data.gt[i]:
      correct += 1
    total += 1

  print correct / total
  return correct / total
