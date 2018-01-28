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
      self.labels = []
 
  # Given a character, returns the character that this "labeler" would respond with
  # based on their accuracy and style matrix
  def getCharacter(self, c):
    
    # Correct character gets "accuracy" percent chance of being selected
    weights = charToOneHot(c) * self.accuracy


    # Every other character has equal "low" chance of being selected
    weights[np.where(weights == 0)] = (1 - self.accuracy) / (NUM_LETTERS - 1)

    # Get the likelihood of outputting each letter considering style
    weights = np.dot(self.style, weights)

    weights *= 100 # So that randint will work

    return chr( getWeightedRand(weights) + 97 )

  # Returns this labeler's answer to the given question
  # In this toy simulation, the correct response to a question
  # is the question repeated. This greatly simplifies the generation
  # of correct and incorrect answers
  def answerQuestion(self, question):
    answer = ""

    for c in question:
      answer += self.getCharacter(c)

    return answer

  def __str__(self):
    ret = "Accuracy: %f\nStyle:\n" % self.accuracy
    ret += str(self.style) + "\n"
    return ret