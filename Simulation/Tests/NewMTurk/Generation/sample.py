from numpy.random import choice
import argparse

from Labeler import *
from Label import *

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Sample data from MTurk dataset.')
  parser.add_argument('infile', help='Filename of dataset to read in')
  # parser.add_argument('outfile', help='Filename to write sampled data to')
  parser.add_argument('n', type=int, help='Number of labels to sample from EACH annotator')
  parser.add_argument('total', type=int, help='Total number of labelers')
  args = parser.parse_args()
  
  for iter in range(100):
    in_fp = open(args.infile, 'r')
    out_fp = open("../data/%d.txt" % iter, 'w')

    metadata1 = in_fp.readline()
    metadata2 = in_fp.readline()
    metadata3 = in_fp.readline()

    Labelers = [Labeler(i) for i in range(args.total)]

    line = in_fp.readline()

    while line != '' and line != '\n':
      line = line.strip().split()
      imgIdx = int(line[0])
      labelerIdx = int(line[1])
      label = int(line[2])
      Labelers[labelerIdx].labels.append(Label(imgIdx, labelerIdx, label))
      line = in_fp.readline()
   
    labels = ""
    numLabels = 0

    for labeler in Labelers:
      sample = choice(labeler.labels, (args.n,), replace=False)
      for lbl in sample:
        labels += str(lbl)
      numLabels += len(sample)

    out_fp.write("%d 25 6 3\n" % (numLabels))
    out_fp.write(metadata2)
    out_fp.write(metadata3)

    out_fp.write(labels)

    # Write out ground truth labels
    out_fp.write(line) # Write out blank line
    line = in_fp.readline()
    while line != '' and line != '\n':
      out_fp.write(line)
      line = in_fp.readline()

    in_fp.close()
    out_fp.close()