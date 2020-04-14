import os
import numpy as np
from matplotlib import pyplot as plt

data_dir = '/home/yuliyang/Downloads/ml_data'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split("\n")
header = lines[0].split(',')

lines = lines[1:]

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# 标准化数据
mean = np.mean(float_data[:20000], axis=0)
float_data -= mean
std = np.std(float_data, axis=0)
float_data /= std


#  data：浮点数数据组成的原始数组，在代码清单 6-32 中将其标准化。
#  lookback：输入数据应该包括过去多少个时间步。
#  delay：目标应该在未来多少个时间步之后。
#  min_index 和 max_index：data 数组中的索引，用于界定需要抽取哪些时间步。这有
# 助于保存一部分数据用于验证、另一部分用于测试。
#  shuffle：是打乱样本，还是按顺序抽取样本。
#  batch_size：每个批量的样本数。
#  step：数据采样的周期（单位：时间步）。我们将其设为 6，为的是每小时抽取一个数据点。
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU

model = keras.models.Sequential()
model.add(GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(Dense(1))
model.compile(optimizer=keras.optimizers.RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=30,
                              epochs=10,
                              validation_data=val_gen,
                              validation_steps=val_steps)
