######## To load the image dataset, and keep it ready to be fed to the model.

from matplotlib.image import imread
import numpy as np
import os

def make_X(directory, case):
        num = len(os.listdir(directory))
        result = np.zeros((num,31,256,3))
        i = 0

        for image in sorted(os.listdir(directory)):
                img = imread(directory+image)
                result[i,:img.shape[0],:img.shape[1],:img.shape[2]] = img[:31,:min(256,img.shape[1]),:3]
                i = i+1

        if case=="test":
                np.save("/home/satvikg/dataset/X_test", result)
        elif case=="train":
                np.save("/home/satvikg/dataset/X_train", result)
        else:
                np.save("/home/satvikg/DeeProkedekt/temp/checkpad", result)
    
        print("Success. Saved np ndarray with shape:")
        print(result.shape)

def make_Y(directory, case):
        num = len(os.listdir(directory))
        Y = np.zeros((num,26))
        m = 0
        for image in sorted(os.listdir(directory)):
                temp = ord(image.split('_')[1][0])
                if temp >= ord('a'):
                        Y[m][temp-ord('a')] = 1
                else:
                        Y[m][temp-ord('A')] = 1
                m += 1

        if case=="test":
                np.save("/home/satvikg/dataset/Y_test", Y)
        elif case=="train":
                np.save("/home/satvikg/dataset/Y_train", Y)
        
        print("Success. Saved np ndarray with shape:")
        print(Y.shape)

#to load it:
#i = np.load("filename")

if __name__=="__main__":
        make_Y("/home/satvikg/dataset/test/", "test")
        make_Y("/home/satvikg/dataset/train/", "train")
        make_X("/home/satvikg/dataset/test/", "test")	
        make_X("/home/satvikg/dataset/train/", "train")	
