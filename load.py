######## To load the image dataset, and keep it ready to be fed to the model.

from matplotlib.image import imread
import numpy as np
import os

path = "/home/satvikg/dataset/train/"
num = len(os.listdir(path))
result = np.zeros((num,31,256,3))
i = 0

for image in os.listdir(path):
        img = imread(path+image)
        result[i,:img.shape[0],:img.shape[1],:img.shape[2]] = img
        i = i+1

np.save("/home/satvikg/dataset/X_train", result)
print("Success. Saved np ndarray with shape:")
print(result.shape)


## To store the numpy arrays 
#to load it:
#i = np.load("filename")


