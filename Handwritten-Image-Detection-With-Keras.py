#!/usr/bin/env python
# coding: utf-8

# ## Handwritten Image Detection with Keras 
# 
# In this project we will work with image data: specifically the famous MNIST and Fashion MNIST data sets.  MNIST data set contains 70,000 images of handwritten digits in grayscale (0=black, 255 = white). Fashion MNIST data set contains 70,000 images of clothing in grayscale (0=black, 255 = white). All the images are 28 pixels by 28 pixels for a total of 784 pixels.  This is quite small by image standards.  Also, the images are well centered and isolated.  This makes this problem solvable with standard fully connected neural nets without too much pre-work. <br><br>
# We will use a Convolutional Neural Network and compare it with a linear neural network. 

# In the first part of this notebook, we will walk you through loading in the data, building a network, and training it.  Then it will be your turn to try different models and see if you can improve performance

# In[1]:


# Preliminaries

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.datasets import fashion_mnist

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's explore the dataset a little bit

# In[2]:


# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train.shape


# In[4]:


#Let's just look at a particular example to see what is inside

x_train[333]  ## Just a 28 x 28 numpy array of ints from 0 to 255


# In[5]:


# What is the corresponding label in the training set?
y_train[333]


# In[6]:


# Let's see what this image actually looks like

plt.imshow(x_train[333], cmap='Greys_r')


# In[7]:


# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[8]:


## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255


# In[9]:


# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[333]  # now the digit k is represented by a 1 in the kth entry (0-indexed) of the length 10 vector


# In[10]:


# We will build a model with two hidden layers of size 512
# Fully connected inputs at each layer
# We will use dropout of .2 to help regularize
model_1 = Sequential()
model_1.add(Dense(64, activation='relu', input_shape=(784,)))
model_1.add(Dropout(0.2))
model_1.add(Dense(64, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(10, activation='softmax'))


# In[11]:


## Note that this model has a LOT of parameters
model_1.summary()


# In[12]:


# Let's compile the model
learning_rate = .001
model_1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=learning_rate),
              metrics=['accuracy'])
# note that `categorical cross entropy` is the natural generalization 
# of the loss function we had in binary classification case, to multi class case


# In[13]:


# And now let's fit.

batch_size = 128  # mini-batch with 128 examples
epochs = 30
history = model_1.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))


# In[14]:


## We will use Keras evaluate function to evaluate performance on the test set

score = model_1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[15]:


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

plot_loss_accuracy(history)


# This is reasonably good performance, but we can do even better!  Next you will build an even bigger network and compare the performance.

# ### -----------------------------------------------------------------------------------------------------------------------------

# ### Keras Layers for CNNs
# - Previously we built Neural Networks using primarily the Dense, Activation and Dropout Layers.
# 
# - Here we will describe how to use some of the CNN-specific layers provided by Keras
# 
# ### (!!!) NOTE THAT WE ARE USING A DIFFERENT MODEL
# 
# #### Conv2D
# 
# ```python
# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
# ```
# 
# A few parameters explained:
# - `filters`: the number of filter used per location.  In other words, the depth of the output.
# - `kernel_size`: an (x,y) tuple giving the height and width of the kernel to be used
# - `strides`: and (x,y) tuple giving the stride in each dimension.  Default is `(1,1)`
# - `input_shape`: required only for the first layer
# 
# Note, the size of the output will be determined by the kernel_size, strides
# 
# #### MaxPooling2D
# `keras.layers.pooling.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)`
# 
# - `pool_size`: the (x,y) size of the grid to be pooled.
# - `strides`: Assumed to be the `pool_size` unless otherwise specified
# 
# #### Flatten
# Turns its input into a one-dimensional vector (per instance).  Usually used when transitioning between convolutional layers and fully connected layers.
# 

# # Exercise
# ### Build your own CNN model
# Use the Keras "Sequential" functionality to build a convolutional neural network `model_2` with the following specifications:
# <br>
# <br>
# Model Architecture:<br>
# We will build the famous LeNet-5 architecutre and measure its performance.
# <br>
#     Convolution -> Relu -> Max pooling -> Convolution -> Relu -> Max pooling -> FC1 -> Relu -> FC2 -> Output(SoftMax)
# <br>
# 
# 1. Convolution1 kernel size: 5(H) x 5(W) x 6(filters), stride = 1, no padding
# 2. Max pooling1 kernel size: 2(H) x 2(W), stride = 2
# 3. Convolution2 kernel size: 5(H) x 5(W) x 16(filters), stride = 1, no padding
# 4. Max pooling2 kernel size: 2(H) x 2(W), stride = 2
# 5. Fully Connected1 size: 120
# 6. Fully Connected1 size: 84
# 7. How many parameters does your model have?  How does it compare with the previous model?
# 8. Train this model for 20 epochs with RMSProp at a learning rate of .001 and a batch size of 128
# 9. Plot the loss and accuracy graph for training the new model 
# 10. Evaluate the model on test data

# To use the LeNet model, we need to do some preprocessing of the data first.

# In[16]:


# Data is currently flattened i.e. m X 784, we need to reshape it back to 28 * 28. To do that we reshape the data.

x_train = np.reshape(x_train, [-1, 28, 28])
x_test = np.reshape(x_test, [-1, 28, 28])
x_train.shape, x_test.shape


# In[17]:


# LeNet requires input of 32 X 32. So, we will pad the train and test images with zeros to increase the size to 32 X 32.

x_train=np.pad(x_train, ((0,0), (2,2), (2, 2)), 'constant')
x_test=np.pad(x_test, ((0,0), (2,2), (2, 2)), 'constant')
x_train.shape, x_test.shape


# In[18]:


# Convolutional model requires input to be of 3 dimensions. We will add a channel dimension to it.

x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])
x_train.shape, x_test.shape


# Write your model below

# In[19]:


#(!!!) UNTESTED MODEL
    #is padding invalid by default?
model2 = Sequential()

#CNN LAYER 1
    #6 filters is the STARTING number
        #'valid' padding is no padding
        #Tuple for strides is the same number all throughout
        #Strides have same dimensions as kernel
model2.add(Conv2D(6, kernel_size=(5,5), strides=(1,1),activation='relu', padding='valid'))
model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#CNN LAYER 2
    #16 filters is the STARTING number
    #'valid' padding is no padding
model2.add(Conv2D(16, kernel_size=(5,5), strides=(1,1),activation='relu', padding='valid'))
model2.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#Fully Connected
    #NEED activation function at every layer
    #Here, we will always use relu
model2.add(Flatten())
    #Just replace Flatten with Dropout for bonus project
model2.add(Dense(120, activation='relu'))
model2.add(Dense(84, activation='relu'))
model2.add(Dense(10, activation='softmax'))
    #Because 10 numbers


# In[20]:


#(!!!) UNTESTED COMPILATION
    #20 epochs 
    #With RMSProp 
    #Learning rate of .001 
    #Batch size of 128
model2.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

"""model2.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])"""

#-----------------------------------------------
#(!!!) WILL CHANGE BATCH SIZE AND EPOCH NUMBER
batch_size = 128  # mini-batch with 128 examples
epochs = 20
history2 = model2.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
#-----------------------------------------------

score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[21]:


## Note that this model has a LOT of parameters
    #Numbers verified to be correct
model2.summary()


# In[22]:


plot_loss_accuracy(history2)


# ### -----------------------------------------------------------------------------------------------------------------------------

# # Fashion MNIST
# We will do the similar things for Fashion MNIST dataset. Fashion MNIST has 10 categories of clothing items:<br>
# 
# | Label | Description | 
# | --- | --- | 
# | 0 | T-shirt/top |
# | 1 | Trouser |
# | 2 | Pullover |
# | 3 | Dress |
# | 4 | Coat |
# | 5 | Sandal |
# | 6 | Shirt |
# | 7 | Sneaker |
# | 8 | Bag |
# | 9 | Ankle boot |

# In[25]:


# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[26]:


x_train[0].shape


# In[27]:


#Let's just look at a particular example to see what is inside

x_train[333]  ## Just a 28 x 28 numpy array of ints from 0 to 255


# In[28]:


# What is the corresponding label in the training set?
y_train[333]


# In[29]:


# Let's see what this image actually looks like

plt.imshow(x_train[333], cmap='Greys_r')


# In[30]:


# this is the shape of the np.array x_train
# it is 3 dimensional.
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[31]:


## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255


# In[32]:


# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[333]  


# ## Reperforming the earlier preprocessing methods

# In[33]:


# Data is currently flattened i.e. m X 784, we need to reshape it back to 28 * 28. To do that we reshape the data.

x_train = np.reshape(x_train, [-1, 28, 28])
x_test = np.reshape(x_test, [-1, 28, 28])
x_train.shape, x_test.shape


# In[34]:


# LeNet requires input of 32 X 32. So, we will pad the train and test images with zeros to increase the size to 32 X 32.

x_train=np.pad(x_train, ((0,0), (2,2), (2, 2)), 'constant')
x_test=np.pad(x_test, ((0,0), (2,2), (2, 2)), 'constant')
x_train.shape, x_test.shape


# In[35]:


# Convolutional model requires input to be of 3 dimensions. We will add a channel dimension to it.

x_train = np.reshape(x_train, [-1, 32, 32, 1])
x_test = np.reshape(x_test, [-1, 32, 32, 1])
x_train.shape, x_test.shape


# I then built a similar convolutional model with a differnet structure, learning rate or number of epochs, etc. that would result in a good model for the dataset

# In[37]:


# write your model here.
"""
--WHAT FACTORS I CAN CHANGE FOR A GOOD MODEL--

# of layers
# of filters (starting number)
#filter size (dimensions)
#size

70% would be GOOD"""


# In[38]:


model3 = Sequential()
model3.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(128, activation='relu'))
model3.add(Dense(10, activation='softmax'))


# In[39]:


model3.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
#-----------------------------------------------
#(!!!) DEBUGGING PURPOSES
    #(!!!) CHANGED BATCH SIZE TO 512
    #(!!!) CHANGED EPOCHS TO 2

#(!!!) WILL CHANGE BATCH SIZE AND EPOCH NUMBER
batch_size = 128  # mini-batch with 128 examples
epochs = 20
history3 = model3.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))
#-----------------------------------------------
score = model3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[40]:


## Note that this model has a LOT of parameters
    #Numbers verified to be correct
model3.summary()


# In[41]:


plot_loss_accuracy(history3)


# In[ ]:




