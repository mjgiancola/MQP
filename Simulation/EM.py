from scipy.optimize import minimize
import numpy as np
import math

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

# Compute P(Z_j | L, S) given S computed in the last MStep
def EStep(data):

  # NOTE: For numerical stability, instead of computing the product in the paper,
  #       we compute the sum of the logs and exponentiate the sum


  # Add log priors
  for x in range(data.numCharacters):
    data.probZ[x] = [np.log(data.priorZ[x][y]) for y in range(data.numImages)]

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

# TODO Write function
def computeQ(data):
  return 2

# TODO Write function
def gradQ(data):
  return 2

def MStep(data):
  x0 = packX(data)

  # Use Conjugate Gradient Ascent to maximize Q (f & df negate returns to maximize)
  res = minimize(f, x0, args=(data,), method='CG', jac=df, options={'maxiter':25,'disp':False})
  unpackX(res.x, data)

# Using minimize, so pass negative Q to maximize
def f(x, data):
  unpackX(x, data)
  return - computeQ(data)

# Again, flip sign since we want to maximize
def df(x, data):
  unpackX(x, data)
  return - gradQ(data)

# Pack style matrices into a vector for CGS
def packX(data):
  x = np.empty(0)
  for i in range(data.numLabelers):
    x = np.append(x, data.style[i].flatten())
  return x

# Unpack style matrices from x
def unpackX(x, data):
  for i in range(data.numLabelers):
    c = data.numCharacters
    mat_size = c ** 2
    style = np.array([x[y] for y in range(i*mat_size, (i*mat_size)+mat_size)])
    data.style[i] = style.reshape((c,c))