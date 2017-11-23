from PIL import Image

import numpy as np
import glob

from random import randint, shuffle

from IPython.display import display

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(filename, loadSize, imageSize, direction=0):
    im = Image.open(filename)
    im = im.resize( (loadSize*2, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2

    # MODIFIED FOR SINGLE CHANNEL!
#    imgA = arr[h1:h2, loadSize+w1:loadSize+w2, :]
#    imgB = arr[h1:h2, w1:w2, :]
    imgA = arr[h1:h2, loadSize+w1:loadSize+w2]
    imgB = arr[h1:h2, w1:w2]

    # mirrors images. Remove
#    if randint(0,1):
#        imgA=imgA[:,::-1]
#        imgB=imgB[:,::-1]
    if direction==0:
        return imgA, imgB
    else:
        return imgB,imgA

def minibatch(dataAB, batchsize, loadSize, imageSize, direction=0):
    length = len(dataAB)
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        
        # if reached end, start new epoch and start again
        if i+size > length:
            shuffle(dataAB)
            i = 0
            epoch+=1        
        # empty list, append with batch images    
        dataA = []
        dataB = []
        for j in range(i,i+size):
            imgA,imgB = read_image(dataAB[j], loadSize, imageSize, direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i+=size

        tmpsize = yield epoch, dataA, dataB  


# For displaying images using matplotlib. Rescale from -1 to 1 to 0 to 1
def raw2display(image):
    return (image+1)/2

# Prints batch data on screen for visulasation
def show_batch(X, imageSize, rows=1):
    #x = raw2display(x)
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')

    int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    display(Image.fromarray(int_X))

def show_batch_grayscale(X, imageSize, rows=1):
    #x = raw2display(x)
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')

    int_X = int_X.reshape(-1,imageSize,imageSize)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize).swapaxes(1,2).reshape(rows*imageSize,-1)
    display(Image.fromarray(int_X))
    
def return_batch_grayscale(X, imageSize, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')

    int_X = int_X.reshape(-1,imageSize,imageSize)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize).swapaxes(1,2).reshape(rows*imageSize,-1)
    return int_X

