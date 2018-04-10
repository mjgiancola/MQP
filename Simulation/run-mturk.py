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
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-n', action='store_true', help='Runs the naive algorithm outlined in our paper')
  parser.add_argument('-m', action='store_true', help='Computes Majority Vote')
  args = parser.parse_args()

  accuracies = []
  cross_entropies = []

  if args.m:
    fp = open("Tests/MechanicalTurk//MV_results.txt", 'w')
  elif args.n:
    fp = open("Tests/MechanicalTurk/BASE_results.txt", 'w')
  elif args.r:
    fp = open("Tests/MechanicalTurk/RSM_results.txt", 'w')
  else:
    fp = open("Tests/MechanicalTurk/DSM_results.txt", 'w')

  for sim in range(100): # Run 100 simulations
    data = init_from_file("Tests/MechanicalTurk/data/%d.txt" % sim, 1, not args.r, True)

    if args.m:
      acc = MV(data)
      result = "Simulation %d: %.2f\n" % (sim, acc*100)
      
    elif args.n:
      acc = naive(data)
      result = "Simulation %d: %.2f\n" % (sim, acc*100)

    else:
      EM(data)
      acc, ce = data.permutedAcc()
      result = "Simulation %d: %.2f %% | %.2f CE\n" % (sim, acc*100, ce)
      cross_entropies.append(ce)
    
    print result
    fp.write(result)
    accuracies.append(acc)

  average_acc = sum(accuracies) / len(accuracies)
  
  if args.m or args.n:
    result = "\nAverage: %.2f %% \n" % (average_acc * 100)

  else:
    average_ce = sum(cross_entropies) / len(cross_entropies)
    result = "\nAverage: %.2f %% | %.2f CE\n" % (average_acc, average_ce)
  
  print result
  fp.write(result)
  fp.close()