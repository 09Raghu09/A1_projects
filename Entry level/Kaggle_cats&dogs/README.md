# Dive into Deep Learning

The goal of the project is finding a data source with images and training a deep learning classifier that works directly on the images.

To understand the concept used "cats and dogs" dataset obtained from https://www.kaggle.com/c/dogs-vs-cats/data. The method which was used was to build a CNN model in Tensorflow 2.0. Here used im.resize() to resize each image to a standard dimension of l00xl00. Converted train images, train labels and test images from lists to numpy arrays. Reshaped our training labels and converted them into a one-hot encoding and normalized it to reduce the range of pixel values in each image to 0.0- 1.0, to obtain better results from our model. Next by building our model using Tensor flow 2.0. After defining our model architecture then created an object for our model and moved on to define our loss functions, optimizer and metrics. 

Finally used categorical cross entropy as the loss function for obtaining single correct result and Adam as the optimizer to update network weights iterative based in training data. Training was performed for the model for 10 epochs and saved the weights of our model after each epoch. We also reset our training loss and accuracy values for each epoch.
