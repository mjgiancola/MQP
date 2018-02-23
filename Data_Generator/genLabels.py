import numpy as np
import skimage.io

# This script generates labels for the Image Segmentation Tests
if __name__=='__main__':
  
  group = 'people'

  im = skimage.io.imread("../Simulation/Tests/ImageSegmentation/%s/Images/gt_seg.png" % group, as_grey=True)
  _, classIm = np.unique(im, return_inverse=True)

  numLabelers = 10
  numImages = len(classIm)
  numLabels = numLabelers * numImages
  numCharacters = 4

  fp = open("../Simulation/Tests/ImageSegmentation/%s/labels.txt" % group, 'w')
  fp.write("%d %d %d %d\n" % ( numLabels, numLabelers, numImages, numCharacters ) )
  
  fp.write("0 1 2 3\n")            # Write character set to file
  fp.write("0.25 0.25 0.25 0.25\n")# Write priors to file
  
  for i in range(10):
    new_im = skimage.io.imread("../Simulation/Tests/ImageSegmentation/%s/Images/noisy%d.png" % (group, i), as_grey=True)
    _, new_classIm = np.unique(new_im, return_inverse=True)

    for j in range(len(classIm)):
      fp.write("%d %d %d\n" % (j, i, new_classIm[j]))

  fp.write("\n")
  for j in range(len(classIm)):
    fp.write("%d %d\n" % (j, classIm[j]))

  fp.close()
