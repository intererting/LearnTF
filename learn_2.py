import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils

(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(
    num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_label = utils.to_categorical(train_labels)

# 建立模型
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(46, activation='softmax'))
# categorical_crossentropy多分类
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.binary_accuracy])

model.fit(x_train, y_label,
          epochs=20, batch_size=512)

predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
