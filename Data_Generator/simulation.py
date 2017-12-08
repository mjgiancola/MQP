import numpy as np
from Labeler import *

def getAnswers(question, Labelers):
  answers = []
  for labeler in Labelers:
    answers.append(labeler.answerQuestion(question))
  return answers

if __name__ == '__main__':
  # In this toy example, I use a three letter alphabet with the letters 'a', 'b', and 'c'
  # Questions and answers are both two letter strings in this alphabet, where the correct answer to a question is the question itself
  # (This is just for simulation simplicity - obviously this can be changed later)
  
  np.random.seed()

  questions = ["aa"]

  #Labeler1 = Labeler(0.8, np.matrix([ [1,0,0], [0,1,0], [0,0,1] ])) # Fairly accurate, canonical style
  #Labeler2 = Labeler(0.5, np.matrix([ [1,0,0], [0,1,0], [0,0,1] ])) # Low accuracy, canonical style
  #Labeler3 = Labeler(0.9, np.matrix([ [0,1,0], [0,0,1], [1,0,0] ])) # High accuracy, style shuffled
  #Labeler4 = Labeler(0.6, np.matrix([ [0,1,0], [0,0,1], [1,0,0] ])) # Low accuracy, style shuffled
  #Labelers = [Labeler1, Labeler2, Labeler3, Labeler4]

  #           in
  #        a   b   c
  # o  a [ 0.1 0.8 0.1 ]
  # u  b [ 0.1 0.1 0.8 ]
  # t  c [ 0.8 0.1 0.1 ]
  #
  Labeler1 = Labeler(0.9, np.matrix([ [0.1,0.8,0.1], [0.1,0.1,0.8], [0.8,0.1,0.1] ]))
  Labelers = [Labeler1]

  for question in questions:
    answers = getAnswers(question, Labelers)
    print "Answers to: " + question
    print answers

  