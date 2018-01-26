import argparse

# TODO Ensure this matches the names you want to map to/from
label_map = {"Group_1":'a', "Group_2":'b', "Group_3":'c', "":""}

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Convert MTurk results files into our simulation datafile format.\nWARNING: Assumes Approve/Reject columns are empty.')
  parser.add_argument('mturk_file_name', help='Filename to read from')
  parser.add_argument('out_file_name', help='Filename to write result to')
  args = parser.parse_args()

  fp = open(args.mturk_file_name, 'r')
  fp.readline() # Skip header line

  labels = []

  line = fp.readline()
  
 	while line != "" and line != "\n":
  	responses = line.replace('"', '').strip().split(',')[28:]
  	converted = [label_map[lbl] for lbl in responses]
  	labels.append(converted)

  	line = fp.readline()

	# TODO Write labels out to file
	# Remember to look for missing labels, continue