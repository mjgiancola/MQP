from scipy.optimize import check_grad
from time import time
import sys, argparse

# Verifies the correctness of Q and gradQ
def check_gradient(data):
  EStep(data)

  x0 = packX(data)
  print check_grad(f, df, x0, data)

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Infer data labels and style matrices.')
  parser.add_argument('-u', action='store_true', help='Runs in unistochastic mode (SinkProp disabled)')
  parser.add_argument('data', help='Filename of dataset file (formatting information in README)')
  args = parser.parse_args()
  
  # Unistochastic mode (SinkProp disabled)
  if args.u:
    from Unistochastic.Dataset import *
    from Unistochastic.EM import *
  
  # Our model (using SinkProp)
  else:
    from SinkProp.Dataset import *
    from SinkProp.EM import *

  data = Dataset(args.data)
  check_gradient(data)

  # start = time()
  # EM(data)
  # elapsed = time() - start
  
  # data.outputResults()
  # print "Completed in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
  # 