######## To load the image dataset, and keep it ready to be fed to the model.


# convert image to numpy ndarray img (x,31,3)
from matplotlib.image import imread
img = imread('image')
print(type(img))
print(img.shape)

## Zero padding all to the same size
result = np.zeros(b.shape)
result[:a.shape[0],:a.shape[1]] = a
print(result)

## To store the numpy arrays 
import numpy as np
np.save("filename", npndarray)
#to load it:
i = np.load("filename")


