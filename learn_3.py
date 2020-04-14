import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils

(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 测试集上的数据标准化也是使用的训练集上的平均值和方差
test_data -= mean
test_data /= std


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        64,
        activation='relu',
        input_shape=(train_data.shape[1],)))

    model.add(keras.layers.Dense(
        64,
        activation=keras.activations.relu
    ))

    model.add(keras.layers.Dense(
        1
    ))

    # mse 均方误差
    # mae 平均绝对误差（MAE，mean absolute error）。它是预测值
    # 与目标值之差的绝对值。比如，如果这个问题的 MAE 等于 0.5，就表示你预测的房价与实际价
    # 格平均相差 500 美元
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss='mse',
                  metrics=['mae', 'accuracy'])
    return model


# 对于数据集很少的情况,使用K折验证
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=32, verbose=0)
    # print(history.history.keys())
    loss_history = history.history['loss']
    all_mae_histories.append(loss_history)

average_loss_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_loss_history) + 1), average_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
