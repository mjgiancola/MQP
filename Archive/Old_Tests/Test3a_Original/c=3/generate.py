import sys, argparse
import numpy as np
from numpy.random import choice, uniform, permutation

from Labeler import *
from Label import *

NUM_LETTERS = 3 # Size of alphabet

def getLabel(ground_truth, labeler):
  return labeler.answerQuestion(ground_truth)

def getLabels(ground_truth, Labelers):
  labels = []
  # i = 9
  for labeler in Labelers:
    # if ground_truth == 'c' and i == 10:
    #   break
    labels.append(labeler.answerQuestion(ground_truth))
    i += 1
  return labels

if __name__ == '__main__':

  np.random.seed()

  parser = argparse.ArgumentParser(description='Generate test data for simulations.')
  # parser.add_argument('data_file_name', help='Filename to write dataset to')
  parser.add_argument('-d', help='Filename to write description of dataset to (optional)')
  args = parser.parse_args()

  types = ['a', 'b', 'c']
  ground_truths = [choice(types) for i in range(100)]

  Labelers = []
  for i in range(80):
    acc = uniform(.75, 1)
    Labelers.append(Labeler(acc, np.identity(3)))

  for i in range(20):
    acc = uniform(.75, 1)
    Labelers.append(Labeler(acc, permutation(np.identity(3))))

  for i in range(100):
    for j in range(len(ground_truths)):
      labeler = Labelers[i]
      gt = ground_truths[j]
      lbl = labeler.answerQuestion(gt)
      Labelers[i].labels.append(Label(j, i, lbl))

  # path = args.data_file_name
  # fp = open(path, 'w')
  if (args.d != None):
    path = args.d
    desc_fp = open(path, 'w')
    desc_fp.write(raw_input("Enter Dataset Description:") + "\n")
    for l in range(len(Labelers)):
      desc_fp.write( "Labeler %d\n" % l )
      desc_fp.write( str(Labelers[l]) )
    desc_fp.close()

  # TODO Set these things before generating

  numLabelers = len(Labelers)
  numImages = len(ground_truths)
  # numLabels = numLabelers * numImages

  labels = []
  for n in range(10, 110, 10):
    numLabels = numLabelers * n
    fp = open("../Simulation/Tests/Test3a/data/data_%d" % n, 'w')
    fp.write("%d %d %d %d\n" % ( numLabels, numLabelers, numImages, NUM_LETTERS ) )
    fp.write("a b c\n")         # Character set
    fp.write("0.33 0.33 0.33\n")# Equal prior for all letters in character set

    for i in range(100):
      labels = choice(Labelers[i].labels, (n,), replace=False)
      for lbl in labels: fp.write(str(lbl))

  # for i in range(len(ground_truths)):
  #   gt = ground_truths[i]
  #   labels = getLabels(gt, Labelers)

  #   for j in range(len(labels)): # Number of labels may not equal the numLabelers if everyone didn't label it
  #     fp.write("%d %d %d\n" % (i, j, ord(labels[j]) - 97) )

  fp.write("\n")
  for i in range(len(ground_truths)):
    fp.write("%d %d\n" % (i, ord(ground_truths[i]) - 97) )

  fp.close()



# Old stuff

  # ground_truths = ["a"] * 25 + ["b"] * 25 + ["c"] * 25# + ["d"] * 25


  # Labeler2 = Labeler(1, np.matrix([ [.9,.1,0,.1], [.1,.9,0,.1], [0,.1,.9,.1], [.1,0,.1,.9] ]))
  # Labeler3 = Labeler(1, np.matrix([ [.3,.1,.1,.5], [.1,.3,.5,.1], [.5,.1,.3,.1], [.1,.5,.1,.3] ]))
  # Labeler4 = Labeler(1, np.matrix([ [.2,0,.2,.6], [0,.2,.6,.2], [2,.6,0,.2], [.6,.2,.2,0] ]))
  # Labeler5 = Labeler(1, np.matrix([ [.8,.05,.05,.1], [.05,.8,.1,.05], [.1,.05,.8,.05], [.05,.1,.05,.8] ]))
  # Labeler6 = Labeler(1, np.matrix([ [.9,.1,0,.1], [.1,.9,0,.1], [0,.1,.9,.1], [.1,0,.1,.9] ]))
  # Labeler7 = Labeler(1, np.matrix([ [.3,.1,.1,.5], [.1,.3,.5,.1], [.5,.1,.3,.1], [.1,.5,.1,.3] ]))
  # Labeler8 = Labeler(1, np.matrix([ [.2,0,.2,.6], [0,.2,.6,.2], [2,.6,0,.2], [.6,.2,.2,0] ]))

  # Labeler1  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler2  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler3  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler4  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler5  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler6  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler7  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler8  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler9  = Labeler(1, np.matrix([ [1.,0.,0.], [0.,1.,0.], [0.,0.,1.] ]))
  # Labeler10 = Labeler(1, np.matrix([ [1.,0.,0.], [0.,0.,1.], [0.,1.,0.] ]))
  # Labelers = [Labeler1, Labeler2, Labeler3, Labeler4, Labeler5, Labeler6, Labeler7, Labeler8, Labeler9, Labeler10]
