from Label import *

class Dataset():
  def __init__(self, filename):

    fp = open(filename, 'r')

    # Read metadata
    line = fp.readline().strip().split()

    self.numLabels = int(line[0])
    self.numLabelers = int(line[1])
    self.numImages = int(line[2])

    tmp_priorZ1 = float(line[3])
    print "Reading %d labels of %d labelers over %d images for prior P(Z=1) = %f" % (self.numLabels, self.numLabelers, self.numImages, tmp_priorZ1)

    print "Assuming prior on alpha has mean 1 and std 1"
    self.priorAlpha = [1 for i in range(self.numLabelers)]

    print "Assuming prior on beta has mean 1 and std 1."
    self.priorBeta = [1 for i in range(self.numImages)]

    print "Also assuming p(Z=1) is the same for all images."
    self.priorZ1 = [tmp_priorZ1 for i in range(self.numImages)]

    # Read labels
    self.labels = []
    line = fp.readline()
    while line != "":
      line = line.strip().split()
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      self.labels.append(lbl)

      if lbl.label != 0 and lbl.label != 1:
        print "Invalid label value"
        exit()

      line = fp.readline()

    # Empty arrays to store these later
    self.probZ1 = []
    self.probZ0 = []
    self.alpha = []
    self.beta =[]

  def outputResults(self):
    
    i = 0
    for a in self.alpha:
      print "Alpha[%d] = %f" % (i, a)
      i += 1

    i = 0
    for b in self.beta:
      print "Beta[%d] = %f" % (i, b)
      i += 1

    i = 0
    for z in self.probZ1:
      print "P(Z(%d)=1) = %f" % (i, z)
      i += 1
