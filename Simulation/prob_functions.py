import numpy as np
from scipy.optimize import minimize

############## EM FUNCTIONS ##############

# Compute P(Z_j | L, S) given S computed in the last MStep
def EStep(data):

  for x in range(data.numCharacters):
    data.probZ[x] = [np.log(data.priorZ[y]) for y in range(data.numImages)]

  # NOTE: For numerical stability, instead of computing the product in the paper,
  #       we compute the sum of the logs and exponentiate the sum

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label    

    # Compute log probabilities
    for z in range(data.numCharacters):
      # From likelihood function - S_z,l
      data.probZ[z][j] += np.log( data.style[i][z][lij] )

  # Exponentiate and renormalize
  for j in range(data.numImages):
    norm_sum = 0
    
    for z in range(data.numCharacters):
      data.probZ[z][j] = np.exp(data.probZ[z][j])
      norm_sum += data.probZ[z][j]

    for z in range(data.numCharacters):
      data.probZ[z][j] /= norm_sum


def computeQ(data):
  pass

def gradQ(data):
  pass

# TODO Write pack/unpack for style
def MStep(data):
  x0 = packX(data)

  # Use Conjugate Gradient Ascent to maximize Q (df gives negative of gradient to maximize)
  res = minimize(f, x0, args=(data,), method='CG', jac=df, options={'maxiter':25,'disp':False})

  unpackX(res.x, data)

########## MISC PROB FUNCTIONS ##########

def logistic(x):
  return 1.0 / (1 + np.exp(-x))