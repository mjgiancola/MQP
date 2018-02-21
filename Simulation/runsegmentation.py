from time import time
import numpy as np
from PIL import Image
import argparse

from Dataset import *
from EM import *
from util.colors import *

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Run simulation on image segmentation data.')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')
  parser.add_argument('xdim', type=int, help='x dimension of image')
  parser.add_argument('ydim', type=int, help='y dimension of image')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-t', action='store_true', help='Indicates if dataset file contains ground truth labels')
  parser.add_argument('-v', action='store_true', help='Verbose mode (Style Matrices)')
  # NOTE: Verbose mode works different in this script than in simulation.py

  args = parser.parse_args()
  data = init_from_file(args.train_data, 1, not args.r, args.t)

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

      if   lbl ==  0: img[i,j] = TAN
      elif lbl ==  1: img[i,j] = GREY
      elif lbl ==  2: img[i,j] = RED   
      elif lbl ==  3: img[i,j] = ORANGE
      elif lbl ==  4: img[i,j] = YELLOW
      elif lbl ==  5: img[i,j] = GREEN
      elif lbl ==  6: img[i,j] = AQUA
      elif lbl ==  7: img[i,j] = PURPLE
      elif lbl ==  8: img[i,j] = BLACK 
      elif lbl ==  9: img[i,j] = PINK
      else:
        print "Uh oh. I didn't plan for this"

  output = Image.fromarray(img, 'RGB')
  if args.r:
    output.save('RSM_output.png')
  else:
    output.save('DSM_output.png')
