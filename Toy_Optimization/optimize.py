import numpy as np
from time import time

SINK_ITER = 30 # Number of Sinkhorn Normalizations to perform on A
GRAD_ITER = 3  # Number of Sinkhorn Propagation steps to perform
REG = 10**(-3) # Regularization term

# Flags
PRINT_MAT = 0 # Print matrices ?

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

# Perform Sinkhorn Normalization
def sink_norm(m, num_iter=SINK_ITER):
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

# iterations is a list of numpy matrices
# where iterations[0] = exp(A)
#       iterations[1] = row_norm(exp(A))
#       iterations[2] = col_norm(row_norm(exp(A)))
#       ...
#       iterations[n-1] = DSM
def compute_gradient(iterations):
  # Want to ensure we finished with a column normalization
  # That way, we can assume so later when performing SinkProp
  assert(len(iterations) % 2 == 1)

  DSM = iterations[ len(iterations) - 1 ]
  df_dA = gradSOS(DSM)

  n = DSM.shape[0] # Dimension of final gradient

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
    df_dA = np.dot(df_dA, partial)

  # Multiply by gradient of expA
  expA = np.reshape(iterations[0], (n**2,))
  d_expA_dA = np.diag(expA)
  df_dA = np.dot(df_dA, d_expA_dA)

  # Reshape 1xn^2 gradient to nxn
  return np.reshape(df_dA, (n,n))

#############################################

# A simple example to verify correctness
# (Hand written solution in Appendix A of report)
def manual_check():
  A = np.array( [ [2., 5.], [3., 4.] ] )

  # Should be [ 2/7 5/7 ]
  #           [ 3/7 4/7 ]
  #print row_norm(A)

  # Should be [ 2/5 5/9 ]
  #           [ 3/5 4/9 ]f
  #print col_norm(row_norm(A))

  # Should be 1.026
  print "Initial function value: " + str(sumofsquares(col_norm(row_norm(A))))

  iterations = sink_norm(A, 1)
  grad = compute_gradient(iterations)
  A += grad

  print "Function value after one Sinkhorn iteration: " + str(sumofsquares(col_norm(row_norm(A))))
  print iterations[len(iterations)-1]

# Prints graphs related to alpha
def lr_test():
  import matplotlib.pyplot as plt

  alphas = np.arange(0, 100.05, 0.05)

  A = np.array( [ [2., 5.], [3., 4.] ] )
  iterations = sink_norm(A)
  grad = compute_gradient(iterations)

  print grad

  # Shows how goodness changes as a function of alpha
  vals = [sumofsquares(col_norm(row_norm(A+alpha*grad))) for alpha in alphas]

  # Shows the minimum value of A as a function of alpha
  vals2 = [np.amin(A+alpha*grad) for alpha in alphas]

  for i in range(len(alphas)):
    if vals2[i] < 0:
      print "Breaking Point: " + str(alphas[i])
      break

  plt.plot(alphas, vals2)
  plt.plot(alphas, np.zeros(alphas.shape))
  plt.show()


# Max value of sum of squares for nxn matrix is n for identity matrix (or any permutation)
if __name__=='__main__':

  start = time()

  # Set to 1 to run learning rate tests
  if 0:
    lr_test()
    exit()

  n = 2 # Dimension of Parameterizing Matrix
  A = np.abs( np.random.randn(n,n) ) + 1 # Initialize a random A

  # Perform Sinkhorn Normalization on exp(A)
  # (Using exp(A) will enforce strict positivity)
  iterations = sink_norm( np.exp(A) + REG )

  alpha = 1000 # Learning rate
  last_sum = np.inf
  curr_sum = sumofsquares(iterations[len(iterations)-1])
  
  if PRINT_MAT:
    print "Original Matrix:"
    print A
  print "Original Sum: " + str(curr_sum) + "\n"

  i=0
  while i < GRAD_ITER:
    gradient = compute_gradient(iterations)
    A += alpha * gradient
    iterations = sink_norm( np.exp(A) + REG )
    i+=1

  end = time()

  print "Resulting Sum: " + str(sumofsquares(iterations[len(iterations)-1]))
  
  if PRINT_MAT:
    print "Resulting DSM:"
    print iterations[len(iterations)-1]
    print "Parameterizing Matrix:"
    print iterations[0]
  
  print "Total Execution Time: " + str(end-start) + " seconds."