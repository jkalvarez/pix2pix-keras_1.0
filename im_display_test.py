# plotting
import matplotlib.pyplot as plt
import matplotlib.image as im

import numpy as np


img = im.imread('./datasets/air_rebar_logit/train/1.jpg')
#img = np.float32(img)
#img = np.array(img)/255*2-1
plt.ion()
plt.imshow(img)
plt.show()

