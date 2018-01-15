from scipy.optimize import check_grad
from time import time
import sys, argparse

from TrainDataset import *
from TestDataset import *
from EM import *

# Verifies the correctness of Q and gradQ
def check_gradient(data):
  EStep(data)

  x0 = packX(data)
  print check_grad(f, df, x0, data)

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Infer data labels and style matrices.')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')
  parser.add_argument('-t', help='Filename of test dataset')
  parser.add_argument('-v', help='Verbose mode')

  args = parser.parse_args()
  
  data = TrainDataset(args.train_data, 1, args.r)
  
  # Uncomment to confirm correctness of Q and gradQ
  # check_gradient(data)

  # Train the model
  start = time()
  EM(data)
  elapsed = time() - start
  
  print "Completed training in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
  if args.v: data.outputResults()

  if args.t != None:
  	test_data = TestDataset(args.t, data, args.r)

  	EStep(test_data)

  	acc = test_data.percent_correct()
  	print "Percent Correct on Test Data: " + str(acc * 100) + "%"
