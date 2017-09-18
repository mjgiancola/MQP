import numpy as np
import random

from util import *

class Labeler:

  # For now, style is limit ed to a doubly-stochastic matrix of the following form:
  # [ a,   1-a ]
  # [ 1-a, a   ]
  def __init__(self, accuracy, style):
      # For now I'm using [0,1] accuracy. (-inf, inf) would be possible if necessary
      # but for several reasons this interval is easier for now
      self.accuracy = accuracy # Float between 0 and 1; 1 = perfect accuracy, 0 = always incorrect
      self.style = style # Bistochastic matrix which represents the style transfer from the norm
    
  # Transforms answer by the labeler's style matrix
  # Matrix assumed to be doubly-stochastic with values 1 and 0 only
  def transformByStyle(self, answer):
    new_answer = ""
    for c in answer:
      one_hot = charToOneHot(c).reshape((NUM_LETTERS,1))
      converted_one_hot = self.style * one_hot  
      new_answer += oneHotToChar(converted_one_hot)
    return new_answer

  # Returns this labeler's answer to the given question
  # In this toy simulation, the correct response to a question
  # is the question repeated. This greatly simplifies the generation
  # of correct and incorrect answers
  def answerQuestion(self, question):
    answer = ""

    for c in question:
      answer += getCharacter(c, self.accuracy)

    return self.transformByStyle(answer)
