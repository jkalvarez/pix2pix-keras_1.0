# Load externals
from models import unet
import utils
from keras import optimizers, losses

# plotting
import matplotlib.pyplot as plt
import matplotlib.image as im

import numpy as np


# Deets and parameters
nc_in = 1
nc_out = 1
ngf = 64

loadSize = 286
imageSize = 128
batchSize = 16
lrG = 2e-4

model = unet(imageSize, nc_in, nc_out, ngf)
model.summary()

# use (model.trainable = False) to freeze weights

# test base unet without discriminator version first

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])
#model.fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)  # starts training

img = im.imread('./datasets/facades/train/1.jpg')
img = np.float32(img)
img = np.array(img)/255*2-1
plt.ion()
plt.imshow(img)

#import matplotlib
#matplotlib.get_backend()
