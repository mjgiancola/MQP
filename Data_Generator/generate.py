import sys, argparse
import numpy as np

from Labeler import *

def getLabels(ground_truth, Labelers):
  labels = []
  for labeler in Labelers:
    labels.append(labeler.answerQuestion(ground_truth))
  return labels

if __name__ == '__main__':
  
  np.random.seed()

  parser = argparse.ArgumentParser(description='Generate test data for simulations.')
  parser.add_argument('-t', action='store_true',
                      help='Generates a test dataset (includes ground truths) (default = train dataset)')
  parser.add_argument('data_file_name', help='Filename to write dataset to')
  parser.add_argument('-d', help='Filename to write description of dataset to (optional)')
  args = parser.parse_args()

  ground_truths = ["d"] * 25 # ["a"] * 25 + ["b"] * 25 + ["c"] * 25 # + ["d"] * 10


  path = "out/" + args.data_file_name
  fp = open(path, 'w')
  if (args.d != None):
    path = "out/" + args.d
    desc_fp = open(path, 'w')
    desc_fp.write(raw_input("Enter Dataset Description:"))
    desc_fp.close()

  # TODO Set these things before generating
  Labeler1 = Labeler(0.8, np.matrix([ [.8,.05,.05,.1], [.05,.8,.1,.05], [.1,.05,.8,.05], [.05,.1,.05,.8] ]))
  Labeler2 = Labeler(0.8, np.matrix([ [.9,.1,0,.1], [.1,.9,0,.1], [0,.1,.9,.1], [.1,0,.1,.9] ]))
  Labeler3 = Labeler(0.8, np.matrix([ [.3,.1,.1,.5], [.1,.3,.5,.1], [.5,.1,.3,.1], [.1,.5,.1,.3] ]))
  Labeler4 = Labeler(0.8, np.matrix([ [.2,0,.2,.6], [0,.2,.6,.2], [2,.6,0,.2], [.6,.2,.2,0] ]))
  Labeler5 = Labeler(0.8, np.matrix([ [.8,.05,.05,.1], [.05,.8,.1,.05], [.1,.05,.8,.05], [.05,.1,.05,.8] ]))
  Labeler6 = Labeler(0.8, np.matrix([ [.9,.1,0,.1], [.1,.9,0,.1], [0,.1,.9,.1], [.1,0,.1,.9] ]))
  Labeler7 = Labeler(0.8, np.matrix([ [.3,.1,.1,.5], [.1,.3,.5,.1], [.5,.1,.3,.1], [.1,.5,.1,.3] ]))
  Labeler8 = Labeler(0.8, np.matrix([ [.2,0,.2,.6], [0,.2,.6,.2], [2,.6,0,.2], [.6,.2,.2,0] ]))
  Labelers = [Labeler1, Labeler2, Labeler3, Labeler4, Labeler5, Labeler6, Labeler7, Labeler8]
  numCharacters = 4

  numLabelers = len(Labelers)
  numImages = len(ground_truths)
  numLabels = numLabelers * numImages
  fp.write("%d %d %d" % ( numLabels, numLabelers, numImages ) )
  if args.t:
    fp.write("\n")
  else:
    fp.write(" %d\n" % numCharacters)

  # TODO Set these before generating
  if not args.t: # Only for training
    fp.write("a b c d\n")            # Character set
    fp.write("0.25 0.25 0.25 0.25\n")# Equal prior for all letters in character set

  for i in range(len(ground_truths)):
    gt = ground_truths[i]
    labels = getLabels(gt, Labelers)

    for j in range(numLabelers):
      fp.write("%d %d %d\n" % (i, j, ord(labels[j]) - 97) )
  
  if args.t:
    fp.write("\n")
    for i in range(len(ground_truths)):
      fp.write("%d %d\n" % (i, ord(ground_truths[i]) - 97) )

  fp.close()