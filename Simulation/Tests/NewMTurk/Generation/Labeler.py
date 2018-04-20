import numpy as np
import random

# Necessary so that, regardless of which labelers are sampled,
# labelers are in range(0,numLabelers)
global_idx = 0

class Labeler:

  def __init__(self, labelerIdx):
    self.labelerIdx = labelerIdx
    self.labels = []

  def addLabel(self, lbl):
    self.labels.append(lbl)

  def __str__(self):
    global global_idx
    ret = ""
    for l in self.labels:
      l.labelerId = ( global_idx % 3 )
      ret += str(l)
    global_idx += 1
    return ret

  def __len__(self):
    return len(self.labels)