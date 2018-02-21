import numpy as np
import skimage.io
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    im = skimage.io.imread("../ADE20K/gt_seg.png", as_grey=True)
    _, classIm = np.unique(im, return_inverse=True)
    classIm = classIm.reshape(im.shape)
    modifiedIm = bleed(classIm)
    plt.imshow(np.hstack((classIm, modifiedIm, modifiedIm - classIm))), plt.show()
