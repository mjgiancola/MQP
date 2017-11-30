from prob_functions import *

THRESHOLD = 1e-5

def EM(data):

  # Initialize style to prior
  data.style = [data.priorStyle[i] for i in range(data.numLabelers)]

  EStep(data) # Estimate P(Z_j | L, S) given priors on S
  lastQ = computeQ(data) # Compute initial Q value
  MStep(data) # Maximize Q to find optimal values of S

  # Iterate until the threshold is reached
  while True:
    EStep(data)
    Q = computeQ(data)
    MStep(data)

    if (math.fabs((Q - lastQ) / lastQ) < THRESHOLD):
      break
