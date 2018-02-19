BLACK  = [0,     0,   0]
PINK   = [238, 130, 238]
DPINK  = [208,  32, 144]
MAGENTA= [255,   0, 255]
RED    = [255,   0,   0]
SIENNA = [138,  54,  15]
SEPIA  = [94,   38,  18]
ORANGE = [255,  97,   3]
YELLOW = [255, 255,   0]
LIME   = [202, 255, 112]
OLIVE  = [142, 142,  56]
GREEN  = [0,   255,   0]
DGREEN = [69,  139,   0]
AQUA   = [0,   255, 255]
MARINE = [127, 255, 212]
SKYBLUE= [198, 226, 255]
VLBLUE = [240, 248, 255]
LBLUE  = [152, 245, 255]
BLUE   = [0,     0, 255]
DBLUE  = [61,   89, 171]
PURPLE = [131, 111, 255]
DPURPLE= [85,   26, 139]
SAND   = [250, 235, 215]
TAN    = [205, 186, 150]
BROWN  = [139, 131, 120]
YBROWN = [139, 139,   0]
GREY   = [128, 138, 135]
WHITE  = [255, 255, 255]

from time import time
import numpy as np
from PIL import Image
import argparse

from Dataset import *
from EM import *

if __name__=='__main__':

  parser = argparse.ArgumentParser(description='Run simulation on image segmentation data.')
  parser.add_argument('train_data', help='Filename of dataset file (formatting information in README)')
  parser.add_argument('xdim', type=int, help='x dimension of image')
  parser.add_argument('ydim', type=int, help='y dimension of image')
  parser.add_argument('-r', action='store_true', help='Runs in right stochastic mode (SinkProp disabled)')
  parser.add_argument('-v', action='store_true', help='Verbose mode')

  args = parser.parse_args()
  
  data = init_from_file(args.train_data, 1, not args.r, False)

  w, h = args.xdim, args.ydim
  img  = np.zeros((h,w,3), dtype=np.uint8)

  start = time()
  EM(data)
  elapsed = time() - start
  
  print "Completed training in %d minutes and %d seconds\n" % (elapsed / 60, elapsed % 60)
  if args.v: data.outputResults()

  observed = np.argmax(data.probZ, axis=0)

  w, h = args.xdim, args.ydim
  img  = np.zeros((h,w,3), dtype=np.uint8)

  for i in range(h):
    for j in range(w):
      lbl = observed[w*i + j]

      if lbl == 0:
        img[i,j] = BLACK
      elif lbl == 1:
        img[i,j] = RED
      elif lbl == 2:
        img[i,j] = YELLOW
      elif lbl == 3:
        img[i,j] = DGREEN
      elif lbl == 4:
        img[i,j] = AQUA
      elif lbl == 5:
        img[i,j] = TAN
      elif lbl == 6:
        img[i,j] = MARINE
      elif lbl == 7:
        img[i,j] = SEPIA
      elif lbl == 8:
        img[i,j] = BROWN
      else:
        print "Uh oh. I didn't plan for this"

  output = Image.fromarray(img, 'RGB')
  output.save('out_new.png')
