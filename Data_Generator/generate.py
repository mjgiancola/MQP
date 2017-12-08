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

  ground_truths = ["a", "b", "c", "d"] * 25

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
  numLabelers = 2
  Labeler1 = Labeler(0.8, np.matrix([ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ])) # Canonical Style
  Labeler2 = Labeler(0.8, np.matrix([ [0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1] ])) # Flips a and c
  Labelers = [Labeler1, Labeler2]
  fp.write("200 2 100 4\n")        # 200 Labels, 2 Labelers, 100 Images, 4 Letters in Character Set
  fp.write("a b c d")              # Character set
  fp.write("0.25 0.25 0.25 0.25\n")# Equal prior for all letters in character set

  for i in range(len(ground_truths)):
    gt = ground_truths[i]
    labels = getLabels(gt, Labelers)

    for j in range(numLabelers):
      fp.write("%d %d %d\n" % (i, j, ord(labels[j]) - 97) )
  
  fp.close()