import numpy as np
import skimage.io
import itertools
from PIL import Image
from random import shuffle

from colors import *

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

      if   lbl ==  0: img[i,j] = AQUA
      elif lbl ==  1: img[i,j] = GREEN
      elif lbl ==  2: img[i,j] = YELLOW
      elif lbl ==  3: img[i,j] = PURPLE
      # elif lbl ==  4: img[i,j] = YELLOW
      # elif lbl ==  5: img[i,j] = GREEN
      # elif lbl ==  6: img[i,j] = AQUA
      # elif lbl ==  7: img[i,j] = PURPLE
      # elif lbl ==  8: img[i,j] = BLACK 
      # elif lbl ==  9: img[i,j] = PINK
      else:
        print lbl
        print "Uh oh. I didn't plan for this"

  output = Image.fromarray(img, 'RGB')
  output.save(filename)

# This script generates randomly permuted noisy versions of an image
if __name__ == "__main__":
    im = skimage.io.imread("../ADE20K/Generated/gt_seg.png", as_grey=True)
    _, classIm = np.unique(im, return_inverse=True)

    # Map classes to reduce 10 classes to 4
    classIm[classIm==1] = 0
    classIm[classIm==2] = 0
    classIm[classIm==3] = 0
    classIm[classIm==4] = 1
    classIm[classIm==5] = 2
    classIm[classIm==6] = 3
    classIm[classIm==7] = 3
    classIm[classIm==8] = 3
    classIm[classIm==9] = 3
    img = classIm.reshape(im.shape)
    saveImage(img, '../Simulation/Tests/ImageSegmentation/Images/gt_seg.png')

    # Generate set of random permutations of <numCharacters> elements
    perms = list(itertools.permutations(range(4)))
    shuffle(perms)

    for i in range(10):
      permuted = np.empty(classIm.shape)
      perm = perms[i]
      j = 0
      for lbl in classIm:
        permuted[j] = perm[lbl]
        j += 1
      permuted = permuted.reshape(im.shape)
      
      modifiedIm = bleed(permuted, numBleeds = 128)
      saveImage(modifiedIm, '../Simulation/Tests/ImageSegmentation/Images/noisy%d.png' % i)
