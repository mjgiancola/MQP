import sys
import numpy as np

from Labeler import *

def getLabels(ground_truth, Labelers):
  labels = []
  for labeler in Labelers:
    labels.append(labeler.answerQuestion(ground_truth))
  return labels

if __name__ == '__main__':
  
  np.random.seed()

  ground_truths = ["a"] * 25 + ["b"] * 25 + ["c"] * 25 + ["d"] * 25

  if (len(sys.argv) < 2):
    print "Usage: python generate.py <data_file_name> [<desc_file_name>]"
    print "where <data_file_name> is the filename where generated data should be written to"
    print "and   <desc_file_name> is the filename where a description of the data will be written to"
    exit()
  
  path = "out/" + sys.argv[1]
  fp = open(path, 'w')
  if (len(sys.argv) == 3):
    path = "out/" + sys.argv[2]
    desc_fp = open(path, 'w')
    desc_fp.write(raw_input("Enter Dataset Description:"))
    desc_fp.close()

  # TODO Set these things before generating
  numLabelers = 5
  Labeler1 = Labeler(0.8, np.matrix([ [.8,.05,.05,.1], [.05,.8,.1,.05], [.1,.05,.8,.05], [.05,.1,.05,.8] ]))
  Labeler2 = Labeler(0.8, np.matrix([ [.9,.1,0,.1], [.1,.9,0,.1], [0,.1,.9,.1], [.1,0,.1,.9] ]))
  Labeler3 = Labeler(0.8, np.matrix([ [.3,.1,.1,.5], [.1,.3,.5,.1], [.5,.1,.3,.1], [.1,.5,.1,.3] ]))
  Labeler4 = Labeler(0.8, np.matrix([ [.2,0,.2,.6], [0,.2,.6,.2], [2,.6,0,.2], [.6,.2,.2,0] ]))
  Labeler5 = Labeler(0.8, np.matrix([ [.25,.25,.25,.25], [.25,.25,.25,.25], [.25,.25,.25,.25], [.25,.25,.25,.25] ]))
  # TODO Try adversarial - .25 for everything
  Labelers = [Labeler1, Labeler2, Labeler3, Labeler4, Labeler5]
  fp.write("500 5 100 4\n")        # Num Labels, Num Labelers, Num Images, Num Letters in Character Set
  fp.write("a b c d\n")            # Character set
  fp.write("0.25 0.25 0.25 0.25\n")# Equal prior for all letters in character set

  for i in range(len(ground_truths)):
    gt = ground_truths[i]
    labels = getLabels(gt, Labelers)

    for j in range(numLabelers):
      fp.write("%d %d %d\n" % (i, j, ord(labels[j]) - 97) )
  
  fp.close()