import numpy as np

# Computes the row-wise softmax of A
def softmax(A):
  S = np.empty(A.shape)
  row_sums = np.sum(np.exp(A), axis=1)
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      S[i][j] = np.exp(A[i][j]) / row_sums[i]
  return S
  
def softmax_gradient(m, i, j, x, y):
  if i != x:
    return 0

  row_sum = float(np.sum(np.exp(m), axis=1)[x])
  
  term1 = ( int(j==y) * np.exp(m[x][y]) ) / row_sum
  term2 = ( np.exp(m[x][y]) * np.exp(m[x][j]) ) / np.square( row_sum )
  return term1 - term2
