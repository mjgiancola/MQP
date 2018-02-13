from time import time
from scipy.optimize import check_grad
import argparse

from Dataset import *
from EM import *

# Verifies the correctness of Q and gradQ
def check_gradient(data):
  EStep(data)

  x0 = packX(data)
  print check_grad(f, df, x0, data)

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Infer data labels and style matrices.')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-t', action='store_true', help='Indicates if dataset file contains ground truth labels')
  parser.add_argument('-p', action='store_true', help='Checks all permutations of labels when computing accuracy (useful for clustering problems')
  parser.add_argument('-v', action='store_true', help='Verbose mode')

  args = parser.parse_args()
  
  data = init_from_file(args.train_data, 1, not args.r, args.t)

  # Uncomment to confirm correctness of Q and gradQ
  # check_gradient(data)

  # Train the model
  if 1:
    start = time()
    EM(data)
    elapsed = time() - start
    
    print "Completed training in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
    if args.v: data.outputResults()

    if args.t:
      if args.p: acc = data.best_percent_correct(); ce = data.best_cross_entropy()
      else:      acc = data.std_percent_correct();  ce = data.std_cross_entropy()
      print "Percent Correct on Test Data: " + str(acc * 100) + "%"
      print "Cross Entropy: " + str(ce)
