from numpy.random import choice
import argparse

from Labeler import *
from Label import *

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Sample data from MTurk dataset.')
  parser.add_argument('infile', help='Filename of dataset to read in')
  parser.add_argument('outfile', help='Filename to write sampled data to')
  parser.add_argument('n', type=int, help='Number of labelers to sample from')
  parser.add_argument('total', type=int, help='Total number of labelers')
  args = parser.parse_args()

  in_fp = open(args.infile, 'r')
  out_fp = open(args.outfile, 'w')

  # First 3 lines are the same
  for i in range(3): out_fp.write(in_fp.readline())

  Labelers = [Labeler(i) for i in range(args.total)]
  # Labelers[0].labels.append(4)
  # print Labelers[0].labels
  # exit()

  line = in_fp.readline()

  while line != '' and line != '\n':
    line = line.strip().split()
    imgIdx = int(line[0])
    labelerIdx = int(line[1])
    label = int(line[2])
    Labelers[labelerIdx].labels.append(Label(imgIdx, labelerIdx, label))
    line = in_fp.readline()
 
  sample = choice(Labelers, (args.n,), replace=False)
  for labeler in sample:
    out_fp.write(str(labeler))

  in_fp.close()
  out_fp.close()