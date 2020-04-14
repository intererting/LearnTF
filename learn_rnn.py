# RNN原理
# state_t = 0
# for input_t in input_sequence:
#   output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
#   state_t = output_t

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense

# timesteps = 100
# input_features = 32
# output_features = 64
# inputs = np.random.random((timesteps, input_features))
# state_t = np.zeros((output_features,))
# W = np.random.random((output_features, input_features))
# U = np.random.random((output_features, output_features))
# b = np.random.random((output_features,))
# successive_outputs = []
# for input_t in inputs:
#     output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
#     successive_outputs.append(output_t)
#     state_t = output_t
# final_output_sequence = np.stack(successive_outputs, axis=0)
# print(final_output_sequence)

max_features = 10000
maxlen = 500
batch_size = 32

imdb.load_data(num_words=max_features)
(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_features)

# pad_sequences
# np.expand_dims
# 这两个都可以扩充维度
# (25000,500)
input_train = sequence.pad_sequences(input_train,
                                     maxlen=maxlen)

input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, 32))
# ，SimpleRNN 不擅长处理长序列，比如文本
model.add(keras.layers.LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.binary_crossentropy,
    metrics=['acc']
)

history = model.fit(input_train, y_train,
                    epochs=5,
                    batch_size=128)

import matplotlib.pyplot as plt

acc = history.history['acc']
# val_acc = history.history['val_acc']
loss = history.history['loss']
# val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
