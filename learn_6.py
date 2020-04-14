import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils
import os, shutil

# 使用已训练好的模型
base_dir = '/home/yuliyang/Downloads/ml_data/ml_dogs_cats/smaller'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# weights 指定模型初始化的权重检查点。
#  include_top 指定模型最后是否包含密集连接分类器。默认情况下，这个密集连接分
# 类器对应于 ImageNet 的 1000 个类别。因为我们打算使用自己的密集连接分类器（只有
# 两个类别：cat 和 dog），所以不需要包含它。
#  input_shape 是输入到网络中的图像张量的形状。这个参数完全是可选的，如果不传
# 入这个参数，那么网络能够处理任意形状的输入
conv_base = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

batch_size = 20


def extract_features(dictory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        dictory,
        target_size=(150, 150)
        , batch_size=batch_size
        , class_mode='binary'
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = keras.models.Sequential()
model.add(keras.layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
