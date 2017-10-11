import numpy as np

NUM_ITERATIONS = 10
THRESHOLD = 10**(-2)
REG = 10**(-3)

############# OBJECTIVE FUNCTION ############

def sumofsquares(m):
  return np.sum(np.multiply(m,m))

#############################################

##### GRADIENTS FOR SUM OF SQUARES ##########

def grad_elt(m, i, j):
  return 2 * m.item((i,j))

def gradSOS(m):
  grad = np.empty(m.shape)
  for i in range(m.shape[0]):
    for j in range(m.shape[1]):
      grad[i][j] = grad_elt(m, i, j)
  return grad

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

# Compute the partial derivative for a row normalization for matrix m at index p,q
# Defined based on derivation in DeepPermNet (similar to SinkProp paper, easier notation)
def row_grad(m, p, q):
  grad = 0
  l = m.shape[0]

  for j in range(l):
    # Using float casts to ensure floating-pt division
    row_sum = float(np.sum(m, axis=1)[p])
    #print("values", m)
    #print("row sum", row_sum)
    term1 = int(j==q) / row_sum # Is it possible to only sum over one row?
    #print("term1", term1)
    term2 = m[p][j] / np.power(row_sum, 2)
    grad += ( term1 - term2 )
  return grad

# Compute the partial derivative for a col normalization for matrix m at index p,q
def col_grad(m, p, q):
  grad = 0
  l = m.shape[0]

  for i in range(l):
    col_sum = float(np.sum(m, axis=0)[q])
    term1 = int(i==p) / col_sum
    #print("term1", term1)
    term2 = m[i][q] / np.power(col_sum, 2)
    grad += ( term1 - term2 )
  return grad

# iterations is a list of numpy matrices
# where iterations[0] = original matrix
#       iterations[1] = row_norm(original matrix)
#       ...
#       iterations[n] = BSM
def compute_gradient(iterations):
  # Want to ensure we finished with a column normalization
  # That way, we can assume so later when performing SinkProp
  assert(len(iterations) % 2 == 1)

  BSM = iterations[ len(iterations) - 1 ]
  sos_grad = gradSOS(BSM)

  # Iterate back through the row and column normalizations
  for idx in range(len(iterations)-1, 0, -1):
    mat = iterations[idx]
    assert mat.shape[0] == mat.shape[1] # Each matrix should be square (this check prob isn't necessary)

    for i in range(BSM.shape[0]):
      for j in range(BSM.shape[1]):
        if idx % 2 == 0:
          sos_grad[i][j] *= col_grad(mat, i, j)
        else:
          sos_grad[i][j] *= row_grad(mat, i, j)
  return sos_grad

#############################################

def _printlist(name, var):
  print(name)
  for x in var:
    print(x)
  print("")

def manual_check():
  x = np.array( [ [2., 5.], [3., 4.] ] )

  # Should be [ 2/7 5/7 ]
  #           [ 3/7 4/7 ]
  #print row_norm(x)

  # Should be [ 2/5 5/9 ]
  #           [ 3/5 4/9 ]
  #print col_norm(row_norm(x))

  # Should be 1.026
  #print sumofsquares(col_norm(row_norm(x)))

  iterations = sink_norm(x, 1)

  grad = compute_gradient(iterations)
  print grad

# Max value of sum of squares for nxn matrix is n for identity matrix (or any permutation)
if __name__=='__main__':

  manual_check()
  exit()

  x = np.array( [ [8., 3., 2.], [4., 3., 3.], [9., 1., 7.] ] )

  last_sum = np.inf
  iterations = sink_norm(x)

  curr_sum = sumofsquares(iterations[len(iterations)-1])
  i=0
  while (last_sum - curr_sum) > THRESHOLD:
    x += compute_gradient(iterations)

    iterations = sink_norm(x)

    last_sum = curr_sum
    curr_sum = sumofsquares(iterations[len(iterations)-1])

    i+=1

  print "Resulting Matrix:"
  print iterations[len(iterations)-1]
  print "Sum of Squares: " + str(sumofsquares(iterations[len(iterations)-1]))