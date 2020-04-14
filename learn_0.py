import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# a = tf.constant([[4.0, 4.0, 4.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
# b = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
# c = tf.square(a - b)
# mse = tf.reduce_mean(c)
# print(c.numpy())
# print(mse.numpy())

# 将数据转化成one-hot数据模式
# random_value = np.random.randint(3, size=(2, 1))
# print(random_value)
# result = keras.utils.to_categorical(random_value, num_classes=5)
# print(result)

# print(np.array([1, 2, 3, 4]).shape)

# (train_images, train_lables), (test_images, test_lables) = keras.datasets.mnist.load_data()

# 只显示一部分图
# plt.imshow(train_images[:1, 14:, 14:][0])
# plt.show()

# relu函数
# def native_relu(x):
#     assert len(x.shape) == 2
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] = max(x[i, j], 0)
#     return x
#
# 54
a = np.array([[1, -2], [3, 4]])
# print(native_relu(a))
b = np.array([[1, -2], [-3, 4]])

# one-hot算法
test = np.zeros((2, 4))
test[1, (1, 2)] = 1
print(test)
# np中的one-hot算法

# test = np.array([1, 2, 3, 4])
# print(utils.to_categorical(test, num_classes=10))

# np版本relu函数54
# print(np.maximum(a, 0))
# print(np.add(a, b))

# 点积
# print(np.dot(a, b))

# 转置
# print(np.transpose(a))

def vectorize_sequences(sequences, dimension=1000):
    """
    :return: (25000,1000)
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


m_imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = m_imdb.load_data(num_words=1000)
# word_index = m_imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 建立模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(1000,)))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 编译
# binary_accuracy二分
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.binary_crossentropy,
              metrics=[keras.metrics.binary_accuracy])

# 训练集和验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train,
                    epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
epochs = range(1, len(loss_value) + 1)

plt.plot(epochs, loss_value, 'bo', label='Training loss')
plt.plot(epochs, val_loss_value, 'b', label='Valudation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
