from scipy.optimize import minimize
import numpy as np
import math

from Normalization.sinkhorn import *
from Normalization.softmax import *
from util.gradient import *

THRESHOLD = 1e-3

def EM(data):

  EStep(data) # Estimate P(Z_j | L, S) given priors on S
  MStep(data) # Maximize Q to find optimal values of S
  lastQ = computeQ(data) # Compute initial Q value

  # initialQ = lastQ

  # Iterate until the threshold is reached
  i = 0
  while True:
    EStep(data)
    MStep(data)
    Q = computeQ(data)

    diff = math.fabs((Q - lastQ) / lastQ)
    print "Iteration " + str(i) + ": " + str(diff) + " (Q=%f)" % Q
    if (diff < THRESHOLD):
     break

    lastQ = Q
    i += 1
  print ""
  return i+1

# Compute P(Z_j | L, S) given S computed in the last MStep
def EStep(data):

  data.computeStyle()

  # NOTE: For numerical stability, instead of computing the product in the paper,
  #       we compute the sum of the logs and exponentiate the sum

  # Add log priors
  for x in range(data.numCharacters):
    data.probZ[x][:] = np.log(data.priorZ[x][:])

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    # Compute log probabilities
    for z in range(data.numCharacters):
      # From likelihood function - S_z,l
      data.probZ[z][j] += np.log( data.Labelers[i].style[z][lij] )

  # Exponentiate and renormalize
  for j in range(data.numImages):
    norm_sum = 0
    
    for z in range(data.numCharacters):
      data.probZ[z][j] = np.exp(data.probZ[z][j])
      norm_sum += data.probZ[z][j]

    for z in range(data.numCharacters):
      data.probZ[z][j] /= norm_sum

def computeQ(data):

  data.computeStyle()

  n = data.Labelers[0].style.shape[0]
  I = np.identity(n)

  result = 0
  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    for z in range(data.numCharacters):
      S_zl = np.log(data.Labelers[i].style[z][lij])
      p_z = data.probZ[z][j]
      result += S_zl * p_z
      
  # Add the prior on S using squared Frobenius norm
  for i in range(data.numLabelers):
    result -= (float(data.gamma) / n**2) * ( np.sum( (data.Labelers[i].style - I) ** 2 ) )

  return result

# Computes the gradient dQ/dA
def compute_gradient(data):
  if data.isDSM: return compute_gradient_DSM(data)
  else:          return compute_gradient_RSM(data)

def compute_gradient_DSM(data):

  data.computeStyle()

  gradients = np.empty(0) # Array of i entries, 1xn^2 matrices, dQdA_i
  dQ_dS = dQdS(data)      # Array of i entries, 1xn^2 matrices, dQdS_i
  n = data.Labelers[0].style.shape[0] # Dimension of matrices

  for i in range(data.numLabelers):
    iterations = data.Labelers[i].iterations
    dQ_dA = dQ_dS[i]

    # "Back propagate" gradient through the row and column normalizations
    for idx in range(len(iterations)-2, -1, -1):
      mat = iterations[idx]

      # Determine type of gradient (row or col)
      if idx % 2 == 1:
        grad_func = col_grad
      else:
        grad_func = row_grad

      # Compute the partial derivative at this step
      partial = gradient_step(mat, grad_func)
      # Add partial to the product
      dQ_dA = np.dot(dQ_dA, partial)

    # Multiply by gradient of expA
    expA = np.reshape(iterations[0], (n**2,))
    d_expA_dA = np.diag(expA)
    dQ_dA = np.dot(dQ_dA, d_expA_dA)

    gradients = np.append(gradients, dQ_dA)

  return gradients

def compute_gradient_RSM(data):

  data.computeStyle()

  gradients = np.empty(0) # Array of i entries, 1xn^2 matrices, dQdA_i
  dQ_dS = dQdS(data)      # Array of i entries, 1xn^2 matrices, dQdS_i
  n = data.Labelers[0].style.shape[0] # Dimension of matrices

  for i in range(data.numLabelers):
    dQ_dA = dQ_dS[i] 
    mat = data.Labelers[i].A
    partial = gradient_step(mat, softmax_gradient)
    dQ_dA = np.dot(dQ_dA, partial)
    gradients = np.append(gradients, dQ_dA)

  return gradients


# The gradient of Q in terms of S, the style matrix
def dQdS(data):
  dQdS = np.empty(0)
  n = data.numCharacters

  dQdS = np.zeros((data.numLabelers,n*n))

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    for x in range(n):
      dQdS[i][x*n + lij] += data.probZ[x][j] / data.Labelers[i].style[x][lij]

  I = np.identity(n)
  for i in range(data.numLabelers):
    for x in range(n):
      for y in range(n):
        Sxy = data.Labelers[i].style[x][y]
        Ixy = I[x][y]
        dQdS[i][x*n + y] -= ( 2.*float(data.gamma) / (n**2) ) * (Sxy - Ixy)

  return dQdS

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
  return - compute_gradient(data)

# Pack parameterizing matrices into a vector for CGS
def packX(data):
  x = np.empty(0)
  for i in range(data.numLabelers):
    x = np.append(x, data.Labelers[i].style.flatten())
  return x

# Unpack parameterizing matrices from x
def unpackX(x, data):
  for i in range(data.numLabelers):
    c = data.numCharacters
    mat_size = c ** 2
    A = np.array([x[y] for y in range(i*mat_size, (i*mat_size)+mat_size)])
    data.Labelers[i].A = A.reshape((c,c))

