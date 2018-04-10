import argparse
import numpy as np
from numpy.random import choice, uniform, permutation

from Dataset import *
from EM import *
from MV import *
from naive import *

# Run multiple trials of a test
if __name__=='__main__':
  np.random.seed()

  parser = argparse.ArgumentParser(description='Run trials of a test.')
  parser.add_argument('n', type=int, help='Number of labels to sample from each labeler')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-n', action='store_true', help='Runs the naive algorithm')
  parser.add_argument('-m', action='store_true', help='Computes Majority Vote')
  args = parser.parse_args()

  alphabet = "a b c"
  types = alphabet.split()
  numCharacters = len(types)

  accuracies = []
  cross_entropies = []

  for sim in range(100): # Run 100 simulations
    data = init_from_file("Tests/RareClass/data/%d.txt" % sim, 0, not args.r, True)

    if args.m:
      acc = MV(data)
      print "Simulation %d: %.2f" % (sim, acc)

    elif args.n:
      acc = naive(data)
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

  if args.m or args.n:
    print "Average: %.2f" % average_acc    

  else:
    average_ce = sum(cross_entropies) / len(cross_entropies)
    print "Average: %.2f % | %.2f CE" % (average_acc, average_ce)
