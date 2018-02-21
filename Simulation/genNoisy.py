import numpy as np
from numpy.random import permutation
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

from util.colors import *

# im is r x c; each element im[r,c] represents the class 
# index assigned to that location by a particular labeler.
def bleed (im, kernelSize = 7, numBleeds = 64):
    im = im.copy()
    for i in range(numBleeds):
        # Randomly generate a bleeding center location
        (r,c) = (np.random.random(2) * (im.shape[0] - kernelSize, im.shape[1] - kernelSize)).astype(np.int32) + (kernelSize/2, kernelSize/2)
        kernel = np.ones((kernelSize, kernelSize)) * im[r,c]
        #print (r,c)
        im[r-kernelSize/2:r-kernelSize/2+kernelSize, c-kernelSize/2:c-kernelSize/2+kernelSize] = kernel

    return im

# A is ndarray with entries representing class index
# filename is where output will be written to
def saveImage(A, filename):
  h, w = A.shape
  img = np.zeros((h,w,3), dtype=np.uint8)

  for i in range(h):
    for j in range(w):
      lbl = A[i][j]

      if   lbl ==  0: img[i,j] = TAN
      elif lbl ==  1: img[i,j] = GREY
      elif lbl ==  2: img[i,j] = RED   
      elif lbl ==  3: img[i,j] = ORANGE
      elif lbl ==  4: img[i,j] = YELLOW
      elif lbl ==  5: img[i,j] = GREEN
      elif lbl ==  6: img[i,j] = AQUA
      elif lbl ==  7: img[i,j] = PURPLE
      elif lbl ==  8: img[i,j] = BLACK 
      elif lbl ==  9: img[i,j] = PINK
      else:
        print lbl
        exit()
        print "Uh oh. I didn't plan for this"

  output = Image.fromarray(img, 'RGB')
  output.save(filename)

if __name__ == "__main__":
    im = skimage.io.imread("../ADE20K/Generated/gt_seg.png", as_grey=True)
    _, classIm = np.unique(im, return_inverse=True)
    classIm = classIm.reshape(im.shape)
    saveImage(classIm, 'Tests/ImageSegmentation/Images/gt_seg.png')

    for i in range(10):
      # For truly random permutations
      #permuted = (((classIm+1) + randint(2, 8)) % 10)
      permuted = ((classIm + i) % 10)
      modifiedIm = bleed(permuted, numBleeds = 128)
      saveImage(modifiedIm, 'Tests/ImageSegmentation/Images/noisy%d.png' % i)

