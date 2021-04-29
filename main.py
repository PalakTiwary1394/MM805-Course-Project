import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# normalize the dataset (0-1) range
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
# not going to scale y as it represents the label as digits

# basic neural network 3 hidden layers and 1 output layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  # adding conv layer, Flatten is one dimensional, feeding
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # Dense-connect all layers prev/after,units-neurons
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))  # 2 Hidden layers
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))  # 3 Hidden layers
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # output layer, softmax- probability

# compile
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(x_train, y_train, epochs= 3)


loss, accuracy = model.evaluate(x_test,y_test)  # in order loss, accuracy
print(accuracy,"\n", loss)

model.save('digits.model')

# img = cv.imread(f'img/1.jpg')
# img = np.array([img])
# plt.imshow(img[0])
# plt.show()


for x in range(1,10):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))    # invert-font color black, bg-white; otherwise messed with nn
    prediction = model.predict(img)
    print("The result is: ",np.argmax(prediction))    # argmax will give us the index of highest prediction value
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

