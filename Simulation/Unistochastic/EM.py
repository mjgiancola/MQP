from scipy.optimize import minimize
from numpy.linalg import norm as frobenius
import numpy as np
import math

from util.sinkhorn import *
from Labeler import *

THRESHOLD = 1e-3
EPSILON = 0.01

def EM(data):

  EStep(data) # Estimate P(Z_j | L, S) given priors on S
  MStep(data) # Maximize Q to find optimal values of S
  lastQ = computeQ(data) # Compute initial Q value

  # Iterate until the threshold is reached
  i = 0
  while True:
    EStep(data)
    MStep(data)
    Q = computeQ(data)

    diff = math.fabs((Q - lastQ) / lastQ)
    if (diff < THRESHOLD):
     break

    lastQ = Q
    print "Iteration " + str(i) + ": " + str(diff)
    i += 1
  print ""

# Compute P(Z_j | L, S) given S computed in the last MStep
def EStep(data):

  computeStyle(data)

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

  computeStyle(data)

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
      
    # Add the prior on S
    for i in range(data.numLabelers):
      # result -= (float(data.gamma) / n**2) * ( np.sum( (data.Labelers[i].style - I) ** 2 ) )
      result -= (float(data.gamma) / n**2) * ( frobenius(data.Labelers[i].style - I) ** 2 )

  return result

def compute_gradient(data):

  computeStyle(data)

  gradients = np.empty(0) # Array of i entries, 1xn^2 matrices, dQdA_i
  dQ_dS = dQdS(data)      # Array of i entries, 1xn^2 matrices, dQdS_i
  n = data.Labelers[0].style.shape[0] # Dimension of matrices

  for i in range(data.numLabelers):
    dQ_dA = dQ_dS[i]
    
    # mat is input to row_norm
    mat = data.Labelers[i].iterations[1]
    partial = gradient_step(mat, row_grad)
    dQ_dA = np.dot(dQ_dA, partial)

    # mat is input to my_relu
    mat = data.Labelers[i].iterations[0]
    partial = np.zeros(mat.shape) + EPSILON
    partial[np.where(mat>0)] = 1 # Derivative of ReLU
    partial = np.diag(np.ravel(partial)) # Derivative needs to be n^2 by n^2 (and will be 0 when indices don't align)
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
        dQdS[i][x*n + y] -= ( 2.*float(data.gamma) / (n**2) ) * (data.Labelers[i].style[x][y] - I[x][y])

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

# Computes the gradient with respect to m, a matrix
# grad_func is either "row_grad" or "col_grad", which compute the gradient of the row norm or column norm respectively
# TODO This may be vectorizable; it is a non-trivial change as the indices need to be set correctly
def gradient_step(m, grad_func):
  n = m.shape[0]
  grad = np.empty((n**2, n**2))

  # Indices of matrices that affect gradient computation
  i = j = x = y = 0

  # Compute each element of the gradient
  for grad_row in range(n**2):
    for grad_col in range(n**2):

      # Take gradient (either row or col)
      grad[grad_row][grad_col] = grad_func(m, i, j, x, y)

      # Update matrix indices
      if j == (n-1):
        if i == (n-1):

          # Finished a row of the gradient, update x,y
          i = 0
          j = 0

          if y == (n-1):
            if x == (n-1):
              # Should be finished - if not, this will alert the presence of a bug
              i = j = x = y = -1
            else:
              x += 1
              y = 0
          else:
            y += 1
        else:
          i += 1
          j = 0
      else:
        j += 1
  return grad
