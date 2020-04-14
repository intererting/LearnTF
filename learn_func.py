from tensorflow import keras
import numpy as np


# input_tensor = keras.Input(shape=(32,))
# dense_a = keras.layers.Dense(16, activation='relu')(input_tensor)
# dense_b = keras.layers.Dense(4, activation='relu')(dense_a)
# output_tensor = keras.layers.Dense(1)(dense_b)
#
# x_train = np.random.random((5000, 32))
# y_train = 0.1 * np.mean(x_train, axis=1) + 0.1
#
# model = keras.Model(input_tensor, output_tensor)
# model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.RMSprop(),
#               metrics=['mae'])
# history = model.fit(x_train, y_train, epochs=10)
#
# print(history.history['mae'])
# test = np.random.random((1, 32))
# print(np.mean(test, axis=1))
# print(model.predict(test)[0])

def many_to_one():
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    num_samples = 2000
    max_length = 100

    text_input = keras.Input(shape=(None,), dtype='int32', name='text')

    embeded_text = keras.layers.Embedding(
        input_dim=text_vocabulary_size, output_dim=64
    )(text_input)

    encoded_text = keras.layers.LSTM(32)(embeded_text)

    question_input = keras.Input(
        shape=(None,)
        , dtype='int32'
        , name='question'
    )

    embedded_question = keras.layers.Embedding(
        question_vocabulary_size, 32)(question_input)
    encoded_question = keras.layers.LSTM(16)(embedded_question)
    concatenated = keras.layers.concatenate([encoded_text, encoded_question],
                                            axis=-1)
    answer = keras.layers.Dense(answer_vocabulary_size,
                                activation='softmax')(concatenated)

    model = keras.Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    text = np.random.randint(1, text_vocabulary_size,
                             size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size,
                                 size=(num_samples, max_length))
    answers = np.random.randint(answer_vocabulary_size, size=(num_samples))
    answers = keras.utils.to_categorical(answers, answer_vocabulary_size)
    # model.fit([text, question], answers, epochs=10, batch_size=128)
    model.fit({'text': text, 'question': question}, answers,
              epochs=10, batch_size=128)


def one_to_many():
    vocabulary_size = 50000
    num_income_groups = 10
    posts_input = keras.Input(shape=(None,), dtype='int32', name='posts')
    embedded_posts = keras.layers.Embedding(256, vocabulary_size)(posts_input)
    x = keras.layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = keras.layers.MaxPooling1D(5)(x)

    x = keras.layers.Conv1D(256, 5, activation='relu')(x)
    x = keras.layers.Conv1D(256, 5, activation='relu')(x)
    x = keras.layers.MaxPooling1D(5)(x)
    x = keras.layers.Conv1D(256, 5, activation='relu')(x)
    x = keras.layers.Conv1D(256, 5, activation='relu')(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    age_prediction = keras.layers.Dense(1, name='age')(x)
    income_prediction = keras.layers.Dense(num_income_groups,
                                           activation='softmax',
                                           name='income')(x)
    gender_prediction = keras.layers.Dense(1, activation='sigmoid', name='gender')(x)
    model = keras.Model(posts_input,
                        [age_prediction, income_prediction, gender_prediction])

    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse',
                        'income': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'},
                  loss_weights={'age': 0.25,
                                'income': 1.,
                                'gender': 10.})

    # model.fit(posts, {'age': age_targets,
    #                   'income': income_targets,
    #                   'gender': gender_targets},
    #           epochs=10, batch_size=64)


from tensorflow.keras import layers


# Inception V3架构
# def one_many_one():
# branch_a = layers.Conv2D(128, 1,
#                          activation='relu', strides=2)(x)
# branch_b = layers.Conv2D(128, 1, activation='relu')(x)
# branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# branch_c = layers.AveragePooling2D(3, strides=2)(x)
# branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
# branch_d = layers.Conv2D(128, 1, activation='relu')(x)
# branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
# branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# output = layers.concatenate(
#     [branch_a, branch_b, branch_c, branch_d], axis=-1)


# 参差连接,将之前的数据连接到后面的输入层,可以避免信息丢失
# 反向传播是用于训练深度神经网络的主要算法，其工作原理是将来自输出损失的反馈信号
# 向下传播到更底部的层。如果这个反馈信号的传播需要经过很多层，那么信号可能会变得非常
# 微弱，甚至完全丢失，导致网络无法训练。这个问题被称为梯度消失（vanishing gradient）。
# 深度网络中存在这个问题，在很长序列上的循环网络也存在这个问题。在这两种情况下，
# 反馈信号的传播都必须通过一长串操作。我们已经知道 LSTM 层是如何在循环网络中解决这
# 个问题的：它引入了一个携带轨道（carry track），可以在与主处理轨道平行的轨道上传播信
# 息。残差连接在前馈深度网络中的工作原理与此类似，但它更加简单：它引入了一个纯线性
# 的信息携带轨道，与主要的层堆叠方向平行，从而有助于跨越任意深度的层来传播梯
# x = ...
# y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
# y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
# y = layers.MaxPooling2D(2, strides=2)(y)
# residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
# y = layers.add([y, residual])

# def share_layers():
# lstm = layers.LSTM(32)
# left_input = Input(shape=(None, 128))
# left_output = lstm(left_input)
# right_input = Input(shape=(None, 128))
# right_output = lstm(right_input)
# merged = layers.concatenate([left_output, right_output], axis=-1)
# predictions = layers.Dense(1, activation='sigmoid')(merged)
# model = Model([left_input, right_input], predictions)
# model.fit([left_data, right_data], targets)


def share_models():
    """
    y = model(x)
    如果模型具有多个输入张量和多个输出张量，那么应该用张量列表来调用模型。
    y1, y2 = model([x1, x2])
    在调用模型实例时，就是在重复使用模型的权重，正如在调用层实例时，就是在重复使用
    层的权重。调用一个实例，无论是层实例还是模型实例，都会重复使用这个实例已经学到的表示，
    这很直观。"""
    pass
# xception_base = applications.Xception(weights=None,
#                                       include_top=False)
# left_input = Input(shape=(250, 250, 3))
# right_input = Input(shape=(250, 250, 3))
# left_features = xception_base(left_input)
# right_input = xception_base(right_input)
# merged_features = layers.concatenate(
#     [left_features, right_input], axis=-1)
