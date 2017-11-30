
# DATA FILE FORMAT #

# numLabels numLabelers numImages numCharacters
# priorZ1 priorZ2 ... priorZ<numCharacters>
# Image_i Labeler_j Label_ij
# ...

class Dataset():
  def __init__(self, filename):

    fp = open(filename, 'r')

    # Read metadata
    line = fp.readline().strip().split()

    # TODO MIKE DO NOT FORGET THAT YOU CHANGED THE METADATA
    self.numLabels = int(line[0])
    self.numLabelers = int(line[1])
    self.numImages = int(line[2])
    self.numCharacters = int(line[3]) # The number of characters in the alphabet

    # TODO Identity is maybe a good prior?
    self.priorStyle = [np.identity(self.numCharacters) for x in range(self.numLabelers)]

    # Read Z priors
    line = fp.readline().strip().split()

    self.priorZ = []
    # Assuming prior is the same for all images
    for x in range(self.numCharacters):
      self.priorZ.append([ int(line[x]) for y in range(self.numImages) ])

    self.labels = []
    # Read in labels
    line = fp.readline()
    while line != "":
      line = line.strip().split()
      
      # Image ID, Labeler ID, Label
      lbl = Label(int(line[0]), int(line[1]), int(line[2]))
      self.labels.append(lbl)

      line = fp.readline()

    # Empty lists to store these later
    self.probZ = []
    for x in range(self.numCharacters):
      self.probZ.append([])
    self.style = []

  # For a large alphabet this will print a lot - consider piping to a file
  def outputResults(self):
    x = 0
    for s in self.style:
      print "Style[%d]:" % x
      print s
      print ""
      x += 1

    for y in range(self.numCharacters): 
      x = 0
      for z in self.probZ[y]:
        print "P(Z(%d)=%d) = %f" % (x, y, z[x]) # TODO Verify that this is what you mean to say
        print ""
        x += 1