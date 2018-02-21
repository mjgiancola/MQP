import numpy as np
import skimage.io

# This script generates labels for the Image Segmentation Tests
if __name__=='__main__':
  
  im = skimage.io.imread("../Simulation/Tests/ImageSegmentation/Images/gt_seg.png", as_grey=True)
  _, classIm = np.unique(im, return_inverse=True)

  numLabelers = 10
  numImages = len(classIm)
  numLabels = numLabelers * numImages
  numCharacters = 10

  fp = open("../Simulation/Tests/ImageSegmentation/labels.txt", 'w')
  fp.write("%d %d %d %d\n" % ( numLabels, numLabelers, numImages, numCharacters ) )
  
  fp.write("0 1 2 3 4 5 6 7 8 9\n") # Write character set to file
  fp.write("0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\n") # Write priors to file
  
  for i in range(10):
    new_im = skimage.io.imread("../Simulation/Tests/ImageSegmentation/Images/noisy%d.png" % i, as_grey=True)
    _, new_classIm = np.unique(new_im, return_inverse=True)

    for j in range(len(classIm)):
      fp.write("%d %d %d\n" % (j, i, new_classIm[j]))

  fp.write("\n")
  for j in range(len(classIm)):
    fp.write("%d %d\n" % (j, classIm[j]))

  fp.close()
