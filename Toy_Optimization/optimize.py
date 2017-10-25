import numpy as np

NUM_ITERATIONS = 30
THRESHOLD = 10**(-2)
REG = 10**(-3)

############# OBJECTIVE FUNCTION ############

def sumofsquares(m):
  return np.sum(np.multiply(m,m))

#############################################

###### GRADIENT FOR SUM OF SQUARES ##########

def gradSOS(m):
  grad = np.empty(m.shape)
  np.copyto(grad, m)
  grad *= 2
  assert m.shape[0] == m.shape[1] # Input should be square
  return np.reshape(grad, (1, m.shape[0]**2))

#############################################

############ NORMALIZATION FUNCTIONS ########

def row_norm(m):
  row_sums = m.sum(axis=1)
  return np.true_divide(m, row_sums[:, np.newaxis])

def col_norm(m):
  col_sums = m.sum(axis=0)
  return np.true_divide(m, col_sums)

# Perform Sinkhorn Normalization some number of iterations
def sink_norm(m, num_iter=NUM_ITERATIONS):
  iterations = [m]
  for i in range(num_iter):
    m = row_norm(m)
    iterations.append(m)
    m = col_norm(m)
    iterations.append(m)
  return iterations

#############################################

######## GRADIENTS FOR NORMALIZATION ########

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

# Computes the gradient with respect to m, a matrix
# grad_func is either "row_grad" or "col_grad", which compute the gradient of the row norm or column norm respectively
def gradient_step(m, grad_func):
  n = m.shape[0]
  grad = np.empty((n**2, n**2))

  i = j = x = y = 0

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

# iterations is a list of numpy matrices
# where iterations[0] = original matrix
#       iterations[1] = row_norm(original matrix)
#       ...
#       iterations[n-1] = BSM
def compute_gradient(iterations):
  # Want to ensure we finished with a column normalization
  # That way, we can assume so later when performing SinkProp
  assert(len(iterations) % 2 == 1)

  BSM = iterations[ len(iterations) - 1 ]
  df_dA = gradSOS(BSM)

  n = BSM.shape[0]

  # Iterate back through the row and column normalizations
  for idx in range(len(iterations)-2, -1, -1):
    mat = iterations[idx]

    # Determine type of gradient (row or col)
    if idx % 2 == 1:
      grad_func = col_grad
    else:
      grad_func = row_grad

    partial = gradient_step(mat, grad_func)

    df_dA = np.dot(df_dA, partial)

  return np.reshape(df_dA, (n,n))

#############################################

# A simple example to verify correctness
# (Hand written solution in Appendix A of report)
def manual_check():
  x = np.array( [ [2., 5.], [3., 4.] ] )

  # Should be [ 2/7 5/7 ]
  #           [ 3/7 4/7 ]
  #print row_norm(x)

  # Should be [ 2/5 5/9 ]
  #           [ 3/5 4/9 ]f
  #print col_norm(row_norm(x))

  # Should be 1.026
  print "Initial function value: " + str(sumofsquares(col_norm(row_norm(x))))

  iterations = sink_norm(x, 1)
  grad = compute_gradient(iterations)
  x += grad

  print "Function value after one Sinkhorn iteration: " + str(sumofsquares(col_norm(row_norm(x))))

# Max value of sum of squares for nxn matrix is n for identity matrix (or any permutation)
if __name__=='__main__':

  x = np.array( [ [2., 5.], [3., 4.] ] )

  iterations = sink_norm(x)

  lr = 15 # Learning rate
  last_sum = np.inf
  curr_sum = sumofsquares(iterations[len(iterations)-1])
  print "Original Sum: " + str(curr_sum)

  i=0
  while i < 1000:
  #while (last_sum - curr_sum) > THRESHOLD: # Sums are too close together for this to be reasonable
    x += lr * compute_gradient(iterations)

    iterations = sink_norm(x)

    last_sum = curr_sum
    curr_sum = sumofsquares(iterations[len(iterations)-1])

    i+=1

  print "Resulting Sum: " + str(sumofsquares(iterations[len(iterations)-1]))
  print "Resulting DSM:"
  print iterations[len(iterations)-1]
  print "Parameterizing Matrix:"
  print iterations[0]