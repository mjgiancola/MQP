from time import time
import numpy as np
from PIL import Image
import argparse

from Dataset import *
from EM import *
from MV import *
from naive import *
from util.colors import *

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Run simulation on image segmentation data.')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')
  parser.add_argument('xdim', type=int, help='x dimension of image')
  parser.add_argument('ydim', type=int, help='y dimension of image')
  
  # Alternate algorithms (i.e. NOT PICA)
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-n', action='store_true', help='Runs the naive algorithm outlined in our paper')
  parser.add_argument('-m', action='store_true', help='Computes Majority Vote')

  # Options for any algorithm (these have no effect when -n is selected)
  parser.add_argument('-t', action='store_true', help='Indicates if dataset file contains ground truth labels')
  parser.add_argument('-v', action='store_true', help='Verbose mode (Style Matrices)')
  # NOTE: Verbose mode works different in this script than in simulation.py

  args = parser.parse_args()
  data = init_from_file(args.train_data, 0, not args.r, args.t)

  # Compute majority vote
  if   args.n: acc = naive(data); print "Percent Correct: " + str(acc * 100) + "%";
  elif args.m: acc = MV(data);    print "Percent Correct: " + str(acc * 100) + "%";
  else:
    # Run our model
    start = time()
    EM(data)
    elapsed = time() - start

    print "Completed training in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
    acc, ce = data.permutedAcc();
    print "Percent Correct: " + str(acc * 100) + "%"
    print "Cross Entropy: " + str(ce)

    if args.v: data.outputStyles()

  observed = np.argmax(data.probZ, axis=0)

  w, h = args.xdim, args.ydim
  img  = np.zeros((h,w,3), dtype=np.uint8)

  for i in range(h):
    for j in range(w):
      lbl = observed[w*i + j]

      if   lbl ==  0: img[i,j] = AQUA
      elif lbl ==  1: img[i,j] = GREEN
      elif lbl ==  2: img[i,j] = YELLOW
      elif lbl ==  3: img[i,j] = PURPLE
      else:
        print "Uh oh. I didn't plan for this"

  output = Image.fromarray(img, 'RGB')
  if args.m:
    output.save('MV_output.png')
  elif args.n:
    output.save('Naive_output.png')
  elif args.r:
    output.save('RSM_output.png')
  else:
    output.save('DSM_output.png')
