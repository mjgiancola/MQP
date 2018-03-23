import argparse
import numpy as np
from numpy.random import choice, uniform, permutation

from GeneratingLabeler.Labeler import Labeler
from Dataset import *
from EM import *
from MV import *

# Run multiple trials of a test
if __name__=='__main__':
  np.random.seed()

  parser = argparse.ArgumentParser(description='Run trials of a test.')
  parser.add_argument('n', type=int, help='Number of labels to sample from each labeler')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-m', action='store_true', help='Computes Majority Vote')
  args = parser.parse_args()

  alphabet = "a b c"
  types = alphabet.split()
  numCharacters = len(types)

  accuracies = []
  cross_entropies = []

  for sim in range(100): # Run 100 simulations
    ground_truths = [choice(types, p=[.5, .45, .05]) for i in range(100)]

    Labelers = []
    for i in range(100):
      acc = uniform(.75, 1)
      Labelers.append(Labeler(acc, permutation(np.identity(numCharacters))))

    for i in range(100):
      for j in range(len(ground_truths)):
        labeler = Labelers[i]
        gt = ground_truths[j]
        lbl = labeler.answerQuestion(gt)
        Labelers[i].labels.append(Label(j, i, ord(lbl) - 97))

    numLabelers = len(Labelers)
    numImages = len(ground_truths)
    gamma = 1
    prior = 1. / len(types) # Equal for all letters in character set

    labels = np.empty(0)
    n = args.n # In paper, we set n=10
    numLabels = numLabelers * n
    
    for i in range(100):
      labels = np.append(labels, choice(Labelers[i].labels, (n,), replace=False))

    data = init_for_trials(numLabels, numLabelers, numImages, numCharacters, gamma, alphabet, prior, labels, not args.r)
    data.gt = np.array(ground_truths)

    if args.m:
      acc = MV(data)
      print "Simulation %d: %.2f" % (sim, acc)

    else:
      EM(data)
      acc = data.best_percent_correct()
      ce = data.best_cross_entropy()
      print "Simulation %d: %.2f % | %.2f CE" % (sim, acc, ce)
      cross_entropies.append(ce)

    accuracies.append(acc)

  average_acc = sum(accuracies) / len(accuracies)
  print"---"

  if args.m:
    print "Average: %.2f" % average_acc    
  
  else:
    average_ce = sum(cross_entropies) / len(cross_entropies)
    print "Average: %.2f % | %.2f CE" % (average_acc, average_ce)