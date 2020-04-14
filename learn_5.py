import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow.keras.utils as utils
import os, shutil

original_dataset_dir = '/home/yuliyang/Downloads/ml_data/ml_dogs_cats/train'
base_dir = '/home/yuliyang/Downloads/ml_data/ml_dogs_cats/smaller'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')


def init_datasets():
    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(150, 150, 3)
    ))

    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(keras.layers.MaxPool2D(2, 2))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


def start_train():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    model = build_model()

    model.fit_generator(
        train_generator,
        steps_per_epoch=100,  # 2000个数据,每次迭代器生成20个,所以100次
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save('cats_and_dogs_small_1.h5')


def shift_data():
    """
     rotation_range 是角度值（在 0~180 范围内），表示图像随机旋转的角度范围。
 width_shift 和 height_shift 是图像在水平或垂直方向上平移的范围（相对于总宽
度或总高度的比例）。
 shear_range 是随机错切变换的角度。
 zoom_range 是图像随机缩放的范围。
 horizontal_flip 是随机将一半图像水平翻转。如果没有水平不对称的假设（比如真
实世界的图像），这种做法是有意义的。
 fill_mode是用于填充新创建像素的方法这些新像素可能来自于旋转或宽度/高度
    """
    data_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    keras_image = keras.preprocessing.image

    frames = [os.path.join(train_cats_dir, frame) for frame in os.listdir(train_cats_dir)]
    image_path = frames[0]
    image = keras_image.load_img(image_path, target_size=(150, 150))
    x = keras_image.img_to_array(image)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in data_gen.flow(x, batch_size=1):
        plt.figure(i)
        plt.imshow(keras_image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break
    plt.show()


shift_data()
