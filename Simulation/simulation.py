from time import time
from scipy.optimize import check_grad
import argparse

from Dataset import *
from EM import *
from naive import *

# Verifies the correctness of Q and gradQ
def check_gradient(data):
  EStep(data)

  x0 = packX(data)
  print check_grad(f, df, x0, data)

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Infer data labels and style matrices.')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')

  # Alternate algorithms (i.e. NOT PICA)
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-n', action='store_true', help='Runs the naive algorithm outlined in our paper')
  
  # Options for any algorithm (these have no effect when -n is selected)
  parser.add_argument('-t', action='store_true', help='Indicates if dataset file contains ground truth labels')
  parser.add_argument('-p', action='store_true', help='Checks all permutations of labels when computing accuracy (useful for clustering problems')
  parser.add_argument('-v', action='store_true', help='Verbose mode (Styles Matrices + probZ)')

  args = parser.parse_args()
  
  data = init_from_file(args.train_data, 0, not args.r, args.t or args.n)

  # Uncomment to confirm correctness of Q and gradQ
  # check_gradient(data)

  # Run the naive (baseline) algorithm
  if args.n:
    print naive(data)

  # Train our model
  else:
    start = time()
    EM(data)
    elapsed = time() - start
    
    print "Completed training in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
    if args.v: data.outputResults()

    if args.t:
      if args.p: acc, ce = data.permutedAcc();
      else:      acc = data.std_percent_correct();  ce = data.std_cross_entropy()
      print "Percent Correct: " + str(acc * 100) + "%"
      print "Cross Entropy: " + str(ce)
