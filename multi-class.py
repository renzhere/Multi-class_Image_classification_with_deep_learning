import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/gdrive')


import os
import numpy as np
from PIL import Image
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def model_M1():
    input_shape = (32, 32, 3)
    model = Sequential()
    
    model.add(Conv2D(16, (5, 5), padding = "same", input_shape = input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(16, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Conv2D(32, (3, 3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from glob import glob


def main():

  ((trainX, trainY), (testX, testY)) = tf.keras.datasets.cifar10.load_data()
  print('shape of input:', trainX.shape)
  
  NUM_EPOCHS = 20
  trainX = trainX.astype("uint8")/ 255.0
  testX = testX.astype("uint8")/ 255.0
    
  print("Compiling model...")
  opt = tf.keras.optimizers.SGD(lr=0.01)
  model1 = model_M1()
  
  model1.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
  
   
  print (model1.summary())
    
  print("Training network ModelM1..... ", )
  H1 = model1.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=64, epochs=NUM_EPOCHS, validation_split=0.2)
    
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["loss"], label="train_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["acc"], label="train_acc")
  plt.plot(np.arange(0, NUM_EPOCHS), H1.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.show()
  
  
  
main()
