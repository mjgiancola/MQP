import numpy as np
from Labeler import *

def getLabels(gt, Labelers):
  labels = []
  for labeler in Labelers:
    labels.append(labeler.getLabel(gt))
  return labels

if __name__ == '__main__':
  # In this toy example, I use a three letter alphabet with the letters 'a', 'b', and 'c'
  # Questions and answers are both two letter strings in this alphabet, where the correct answer to a question is the question itself
  # (This is just for simulation simplicity - obviously this can be changed later)
  
  np.random.seed()

  ground_truths = ["a", "b", "c"]

  #Labeler1 = Labeler(0.8, np.matrix([ [1,0,0], [0,1,0], [0,0,1] ])) # Fairly accurate, canonical style
  #Labeler2 = Labeler(0.5, np.matrix([ [1,0,0], [0,1,0], [0,0,1] ])) # Low accuracy, canonical style
  #Labeler3 = Labeler(0.9, np.matrix([ [0,1,0], [0,0,1], [1,0,0] ])) # High accuracy, style shuffled
  #Labeler4 = Labeler(0.6, np.matrix([ [0,1,0], [0,0,1], [1,0,0] ])) # Low accuracy, style shuffled
  #Labelers = [Labeler1, Labeler2, Labeler3, Labeler4]

  #           in
  #        a   b   c
  # o  a [ 0.1 0.8 0.1 ]
  # u  b [ 0.1 0.1 0.8 ]
  # t  c [ 0.8 0.1 0.1 ]
  #
  Labeler1 = Labeler(0.9, np.matrix([ [0.1,0.8,0.1], [0.1,0.1,0.8], [0.8,0.1,0.1] ]))
  Labelers = [Labeler1]

  for gt in ground_truths:
    given_labels = getLabels(gt, Labelers)
    print "Labels for: " + gt
    print given_labels

  