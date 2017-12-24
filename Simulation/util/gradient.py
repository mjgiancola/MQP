import numpy as np

# Computes the gradient with respect to m, a matrix
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

