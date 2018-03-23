import numpy as np

SINK_ITER = 50

# Perform Sinkhorn Normalization
# Assumes m is strictly positive (so in practice, generally pass exp(A) as m)
def sink_norm(m, num_iter=SINK_ITER, iter_list=True):
  iterations = [m]

  for i in range(num_iter):
    m = row_norm(m)
    iterations.append(m)

    m = col_norm(m)
    iterations.append(m)

  if iter_list:
    return iterations
  else:
    return iterations[len(iterations)-1]

def row_norm(m):
  row_sums = m.sum(axis=1)
  return np.true_divide(m, row_sums[:, np.newaxis])

def col_norm(m):
  col_sums = m.sum(axis=0)
  return np.true_divide(m, col_sums)

def row_grad(m, i, j, x, y):
  # Gradient is zero if the rows are different
  if i != x:
    return 0

  row_sum = float(np.sum(m, axis=1)[x])
  term1 = int(j==y) / row_sum
  term2 = m[x][y] / np.power(row_sum, 2)

  return term1 - term2

def col_grad(m, i, j, x, y):
  # Gradient is zero if the columns are different
  if j != y:
    return 0

  col_sum = float(np.sum(m, axis=0)[y])
  term1 = int(i==x) / col_sum
  term2 = m[x][y] / np.power(col_sum, 2)

  return term1 - term2
