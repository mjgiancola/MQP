import numpy as np

# Pack alpha and beta into a vector for CGS
def packX(data):
  x = np.empty(0)
  for i in range(data.numLabelers):
    x = np.append(x, data.alpha[i])
  for j in range(data.numImages):
    x = np.append(x, data.beta[j])
  return x

# Unpack alpha and beta from x
def unpackX(x, data):
  for i in range(data.numLabelers):
    data.alpha[i] = x[i]
  for j in range(data.numImages):
    data.beta[j] = x[data.numLabelers + j]
