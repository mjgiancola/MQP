import numpy as np
import random

class Labeler:

  def __init__(self, labelerIdx):
    self.labelerIdx = labelerIdx
    self.labels = []

  def addLabel(self, lbl):
    self.labels.append(lbl)

  def __str__(self):
    ret = ""
    for l in self.labels:
      ret += str(l)
    return ret
