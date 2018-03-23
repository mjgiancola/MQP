import sys, argparse
import numpy as np
from numpy.random import choice, uniform, permutation, rand

sys.path.insert(0, '../util')
sys.path.insert(0, '../../Simulation/Normalization')
from Labeler import *
from Label import *
from sinkhorn import *

if __name__ == '__main__':

  np.random.seed()

  types = ['a', 'b', 'c']
  n = len(types)
  free = (n - 1)**2
  numLabelers = 25
  ground_truths = [choice(types) for i in range(5 * free * numLabelers)]

  Labelers = []

  for i in range(numLabelers):
    acc = uniform(.75, 1)
    style = sink_norm(abs(rand(n,n)), num_iter=100, iter_list=False)
    Labelers.append(Labeler(acc, style, n))

  for i in range(numLabelers):
    for j in range(len(ground_truths)):
      labeler = Labelers[i]
      gt = ground_truths[j]
      lbl = labeler.answerQuestion(gt)
      Labelers[i].labels.append(Label(j, i, lbl))

  numImages = len(ground_truths)
  numSampled = numImages / 2

  labels = []
  for sim in range(50):
    numLabels = numSampled * numLabelers
    fp = open("../../Simulation/Tests/SinkhornAnalysis/data/%d.txt" % sim, 'w')
    fp.write("%d %d %d %d\n" % ( numLabels, numLabelers, numImages, n ) )
    
    # Write character set to file
    for c in types: fp.write("%s " % c)
    fp.write("\n")
    
    # Write priors to file
    prior = 1. / len(types) # Equal for all letters in character set
    for c in types: fp.write("%.2f " % prior)
    fp.write("\n")

    for i in range(numLabelers):
      labels = choice(Labelers[i].labels, (numSampled,), replace=False)
      for lbl in labels: fp.write(str(lbl))

    fp.write("\n")
    for i in range(len(ground_truths)):
      fp.write("%d %d\n" % (i, ord(ground_truths[i]) - 97) )

  fp.close()
