# DeeProkedekt
The second version of Prokedekt, which used Deep Learning to recognize text from textboxes, which are detected from an image using EAST Detector. 

## Tasks:
* [DONE] The [dataset](http://www.robots.ox.ac.uk/~vgg/data/text/) used for training.
* [DONE] Clean the dataset.
* [DONE] Convert images to np arrays of the same size.
* [DONE] Create ground truth file (output layer)
* [DONE] Create train/dev/test sets. 
* [DONE] Make minibatches of train set. 
* [DONE] Learn Keras to build a basic CNN.
* [DONE] Train it and check results on the test set. 
* [DONE] Document the results of different models and choose the best working model. (CNN, RNN, Adam, ResNet, Inception, etc)
* Fine tune hyperparameters.



## Dataset Citation:
@article{Jaderberg14c,
      title={Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1406.2227},
      year={2014}
    }
    
@article{Jaderberg14d,
      title={Reading Text in the Wild with Convolutional Neural Networks},
      author={Jaderberg, M. and Simonyan, K. and Vedaldi, A. and Zisserman, A.},
      journal={arXiv preprint arXiv:1412.1842},
      year={2014}
    } 
