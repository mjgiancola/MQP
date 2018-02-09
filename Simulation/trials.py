import argparse
import numpy as np
from numpy.random import choice, uniform, permutation

from Dataset import *
from EM import *

# Run multiple trials of a test
if __name__=='__main__':
  np.random.seed()

  parser = argparse.ArgumentParser(description='Run trials of a test.')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  args = parser.parse_args()

  accuracies = []
  cross_entropies = []  

  if not args.r:
    fp = open("Tests/Test4/3labelers_100trials/DSM_results.txt", 'w')
  else:
    fp = open("Tests/Test4/3labelers_100trials/RSM_results.txt", 'w')

  for sim in range(100): # Run 100 simulations
    data = init_from_file("Tests/Test4/3labelers_100trials/data/%d.txt" % sim, 1, not args.r, True)
  
    EM(data)

    acc = data.best_percent_correct()
    ce = data.best_cross_entropy()

    result = "Simulation %d: %.2f %% | %.2f CE\n" % (sim, acc*100, ce)
    print result
    fp.write(result)

    accuracies.append(acc)
    cross_entropies.append(ce)

  average_acc = sum(accuracies) / len(accuracies)
  average_ce = sum(cross_entropies) / len(cross_entropies)

  result = "\nAverage: %.2f %% | %.2f CE\n" % (average_acc, average_ce)
  print result
  fp.write(result)
  fp.close()