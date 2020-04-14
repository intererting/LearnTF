import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils
import os, shutil

# 模型可视化
base_dir = '/home/yuliyang/Downloads/ml_data/ml_dogs_cats'
train_dir = os.path.join(base_dir, 'train')
image_path = os.path.join(train_dir, r'cat.1700.jpg')

keras_image = keras.preprocessing.image
image = keras_image.load_img(image_path, target_size=(150, 150))
image_tensor = keras_image.img_to_array(image)
image_tensor = np.expand_dims(image_tensor, axis=0)
image_tensor /= 255
# plt.imshow(image_tensor[0])
# plt.show()

model = keras.models.load_model('cats_and_dogs_small_1.h5')

print(model.summary())
# 取出卷积层
layout_outputs = [layout.output for layout in model.layers[:8]]
activation_model = keras.models.Model(
    inputs=model.input,
    outputs=layout_outputs)

activations = activation_model.predict(image_tensor)
# 获取第一个卷积层的结果
first_layer_activation = activations[0]
# print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
