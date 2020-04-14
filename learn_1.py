import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime


def learn_non_linear_reg():
    x_data = np.linspace(-0.5, 0.5, 2000)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise

    model = keras.models.Sequential([
        keras.layers.Dense(4, activation='tanh'),
        keras.layers.Dense(1)
    ])

    model.compile(loss='logcosh', optimizer='adam')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x_data, y_data, epochs=20, callbacks=[tensorboard_callback])
    # test_data = np.linspace(-0.5, 0.5, 10)[:, np.newaxis]

    # predict_data = model.predict(test_data)
    #
    # plt.figure()
    # plt.scatter(test_data, predict_data)
    # plt.show()


learn_non_linear_reg()


def learn_linear_reg():
    # 生成随机数
    x_data = np.random.rand(1000)
    y_data = 0.1 * x_data + 0.2

    # 构造线性模型
    model = keras.models.Sequential([
        keras.layers.Dense(2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='huber_loss', optimizer='adam')

    model.fit(x_data, y_data, epochs=50)
    test_data = np.random.random(5)

    print(test_data)
    print(model.predict(test_data))
