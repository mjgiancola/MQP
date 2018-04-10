import argparse

# TODO Set this before running
clusters = ["Group_1", "Group_2", "Group_3"]
NUM_LETTERS = len(clusters)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Convert MTurk results files into our simulation datafile format.\nWARNING: Assumes Approve/Reject columns are empty.')
  parser.add_argument('mturk_file_name', help='Filename to read from')
  parser.add_argument('out_file_name', help='Filename to write result to')
  args = parser.parse_args()

  fp = open(args.mturk_file_name, 'r')
  fp.readline() # Skip header line

  labels = []
  missingResponse = 0 # Counts number of missing labels

  line = fp.readline()
  while line != "" and line != "\n":
    responses = line.replace('"', '').strip().split(',')[28:]
    labels.append(responses)

    line = fp.readline()
    missingResponse += responses.count('')

  fp.close()

  fp = open(args.out_file_name, 'w')

  numLabelers = len(labels)
  numImages   = len(labels[0]) # Assumes each labeler answered each question, or has "" for a missing answer
  numLabels = ( numLabelers * numImages ) - missingResponse
  fp.write("%d %d %d %d\n" % ( numLabels, numLabelers, numImages, NUM_LETTERS ) )
  for c in clusters: fp.write("%s " % c) # Write answer set
  fp.write("\n")
  fp.write("0.33 0.33 0.33\n") # Equal prior for all letters in character set

  for labeler in range(len(labels)):
    for img in range(len(labels[labeler])):
      lbl = labels[labeler][img]
      if lbl == '': continue
      fp.write("%d %d %d\n" % (img, labeler, clusters.index(lbl)))


  fp.close()