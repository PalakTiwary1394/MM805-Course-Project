import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import tensorflow as tf

train = pd.read_csv('mnist/train.csv')
test = pd.read_csv('mnist/test.csv')

y_train = train[['label']]
x_train = train.drop(train.columns[[0]], axis=1)
x_test = test

# Visualizing the data
sample = x_train.iloc[10, :]
sample = sample.values.reshape([28,28])
plt.imshow(sample, cmap='gray')

x_train = np.array(x_train)
x_test = np.array(x_test)

# Reshape the training and test set
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Padding the images by 2 pixels since in the paper input images were 32x32
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Standardization
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
x_train = (x_train - mean_px)/(std_px)

# One-hot encoding the labels
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()
# Layer 1
# Conv Layer 1
model.add(Conv2D(filters = 6,
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (32,32,1)))
# Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
# Layer 2
# Conv Layer 2
model.add(Conv2D(filters = 16,
                 kernel_size = 5,
                 strides = 1,
                 activation = 'relu',
                 input_shape = (14,14,6)))
# Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
# Flatten
model.add(Flatten())
# Layer 3
# Fully connected layer 1
model.add(Dense(units = 120, activation = 'relu'))
# Layer 4
# Fully connected layer 2
model.add(Dense(units = 84, activation = 'relu'))
# Layer 5
# Output Layer
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train ,y_train, steps_per_epoch = 10, epochs = 2)

y_pred = model.predict(x_test)

# Converting one hot vectors to labels
labels = np.argmax(y_pred, axis = 1)

index = np.arange(1, 28001)

labels = labels.reshape([len(labels),1])
index = index.reshape([len(index), 1])

final = np.concatenate([index, labels], axis = 1)

# Prediction csv file
np.savetxt("mnist_2.csv", final, delimiter = " ", fmt = '%s')

# loss, accuracy = model.evaluate(x_test,y_pred)  # in order loss, accuracy
# print(accuracy,"\n", loss)

# print("x: ",x_test,"y_pred: ",y_pred)

