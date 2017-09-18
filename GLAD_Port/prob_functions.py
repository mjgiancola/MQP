import numpy as np
from numpy.random import normal
from scipy.optimize import minimize

from data import *

# Returns the log probability of label l (QUESTION: Why doesn't this match paper? 1 / 1 + e^ab)
# I know exponentiation happens later to reverse log, but why is the result negative?
# Also, why is beta to the e again? Later, in the code for gradientQ, alpha isn't included in the exp at all..
# l      = given label
# z      = "true" label
# alphaI = accuracy of labeler i
# betaJ  = difficulty of image j
def logProbL(l, z, alphaI, betaJ):
  if (z==l):
    return - np.log(1 + np.exp(- np.exp(betaJ) * alphaI))
  else:
    return - np.log(1 + np.exp(np.exp(betaJ) * alphaI))

def logistic(x):
  return 1.0 / (1 + np.exp(-x))

# Re-estimate P(Z|L,alpha,beta)
def EStep(data):
  data.probZ1 = [np.log(data.priorZ1[i]) for i in range(data.numImages)]
  data.probZ0 = [np.log(1 - data.priorZ1[i]) for i in range(data.numImages)]

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label # Get label from labeler i, image j

    # Compute log probabilities
    data.probZ1[j] += logProbL(lij, 1, data.alpha[i], data.beta[j])
    data.probZ0[j] += logProbL(lij, 0, data.alpha[i], data.beta[j])

  # Exponentiate and renormalize
  for j in range(data.numImages):
    data.probZ1[j] = np.exp(data.probZ1[j])
    data.probZ0[j] = np.exp(data.probZ0[j])

    data.probZ1[j] = data.probZ1[j] / (data.probZ1[j] + data.probZ0[j])
    data.probZ0[j] = 1 - data.probZ1[j]
    if (np.isnan(data.probZ1[j])):
      print "Uh oh: data.probZ1 is NAN"
      exit()

import warnings
warnings.filterwarnings('error')
import time

# Compute the auxilary function Q
def computeQ(data):
  # For convenience
  alpha = data.alpha
  beta = data.beta

  Q = 0

  # Start with the expectation of the sum of priors over all images
  for j in range(data.numImages):
    Q += data.probZ1[j] * np.log(data.priorZ1[j])
    Q += data.probZ0[j] * np.log(1 - data.priorZ1[j])

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    # Analytical manipulation for numerical stability
    logSigma = - np.log(1 + np.exp(- np.exp(beta[j]) * alpha[i]))
    if (not np.isfinite(logSigma)):
      logSigma = np.exp(beta[j]) * alpha[i]

    logOneMinusSigma = - np.log(1 + np.exp( np.exp(beta[j]) * alpha[i]))
    if (not np.isfinite(logOneMinusSigma)):
      # For large positive x, -log(1+ exp(x)) = x
      logOneMinusSigma = - np.exp(beta[j]) * alpha[i]

    # This equation can be found on the second page of the full derivation (in the supp. materials)
    Q += data.probZ1[j] * (lij       * logSigma + (1 - lij) * logOneMinusSigma) + \
         data.probZ0[j] * ((1 - lij) * logSigma +      lij  * logOneMinusSigma)

    if (np.isnan(Q)):
      print "Uh oh: Q is NAN"
      exit()

    # QUESTION: I don't totally understand where these come from.. I don't see the corresponding equations in the derivation
    # Add Gaussian (standard normal) prior for alpha
    # for i in range(data.numLabelers):
    #   Q += np.log( normal(alpha[i] - data.priorAlpha[i]))

    # # Add Gaussian (standard normal) prior for beta
    # for j in range(data.numImages):
    #   Q += np.log ( normal(beta[j] - data.priorBeta[j]))

  return Q

def gradientQ(data):
  dQdAlpha = np.empty(0); dQdBeta = np.empty(0);

  for i in range(data.numLabelers):
    dQdAlpha = np.append(dQdAlpha, -(data.alpha[i] - data.priorAlpha[i]))
  for j in range(data.numImages):
    dQdBeta = np.append(dQdBeta, -(data.beta[j] - data.priorBeta[j]))

  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    # QUESTION: Why is this computed for exp(b) * a, not b * a as the paper states?
    sigma = logistic( np.exp(data.beta[j]) * data.alpha[i])

    dQdAlpha[i] += (data.probZ1[j] * (lij - sigma) + data.probZ0[j] * (1 - lij - sigma)) * np.exp(data.beta[j])
    dQdBeta[j]  += (data.probZ1[j] * (lij - sigma) + data.probZ0[j] * (1 - lij - sigma)) * np.exp(data.beta[j]) * data.alpha[i]

  return dQdAlpha, dQdBeta

# CG will minimize, so we pass it the negative of Q
def f(x, data):
  unpackX(x, data)
  print "f"
  print data.alpha
  return - computeQ(data)

def df(x, data):
  unpackX(x, data)
  dQdAlpha, dQdBeta = gradientQ(data)
  print "df"
  print dQdAlpha
  return np.concatenate((-dQdAlpha, -dQdBeta)) # Flip sign since we want to minimize

# Maximize the auxilary function Q
def MStep(data):
  numLabelers = data.numLabelers
  numImages = data.numImages

  x0 = packX(data)

  # Perform Conjugate Gradient to maximize Q
  res = minimize(f, x0, args=(data,), method='CG', jac=df, options={'maxiter':25, 'disp':False})

  unpackX(res.x, data)
