

def MV(data):
  for idx in range(data.numLabels):
    i = data.labels[idx].labelerId
    j = data.labels[idx].imageIdx
    lij = data.labels[idx].label

    data.probZ[lij][j] += 1

  # Compute percent correct
  acc, _ = data.permutedAcc()
  print "Percent Correct: " + str(acc * 100) + "%"