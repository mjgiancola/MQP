import argparse
from time import time
import numpy as np
from numpy.random import choice, uniform, permutation

from Dataset import *
from EM import *
from MV import *

# Run multiple trials of a test
if __name__=='__main__':
  np.random.seed()

  accuracies = []
  cross_entropies = []
  times = []
  steps = []

  for sim in range(50): # Run 50 simulations
    data = init_from_file("Tests/SinkhornAnalysis/data/%d.txt" % sim, 0, True, True)

    start = time()
    num_steps = EM(data)
    elapsed = time() - start
    times.append(elapsed)
    steps.append(num_steps)

    print("Simulation 1:")
    print("Elapsed Time: %d minutes, %d seconds" % (elapsed / 60, elapsed % 60))
    
    acc, ce = data.permutedAcc()
    print "Accuracy: %.2f %% | %.2f CE\n" % (acc*100, ce)
    accuracies.append(acc)
    cross_entropies.append(ce)

  average_acc = sum(accuracies) / len(accuracies)
  average_ce  = sum(cross_entropies) / len(cross_entropies)
  average_time = sum(times) / len(times)
  average_steps = sum(steps) / len(steps)

  print "Average Accuracy: %.2f %%" % average_acc * 100
  print "Average Cross-Entropy: %.2f" % average_ce
  print "Average Time: %d minutes, %d seconds" % (average_time / 60, average_time % 60)
  print "Average Steps: %d" % average_steps
  print ""
