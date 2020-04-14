import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation='relu',
    input_shape=(28, 28, 1)))

model.add(keras.layers.MaxPool2D(2, 2))

model.add(keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation='relu'
))

model.add(keras.layers.MaxPool2D(2, 2))
# 卷积神经网络接收形状为 (image_height, image_width, image_channels)
# 的输入张量（不包括批量维度）。
model.add(keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(
    units=10,
    activation='relu',
))

model.add(keras.layers.Dense(
    units=10,
    activation='softmax'
))

print(model.summary())

# (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
# train_images = train_images.reshape((60000, 28, 28, 1))
# train_images = train_images.astype('float32') / 255
# test_images = test_images.reshape((10000, 28, 28, 1))
# test_images = test_images.astype('float32') / 255
# train_labels = utils.to_categorical(train_labels)
# test_labels = utils.to_categorical(test_labels)
# print(train_labels.shape)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
