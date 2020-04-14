from tensorflow import keras
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding

max_features = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)
print(x_train.shape)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train.shape)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(
    input_dim=10000,
    output_dim=8,
    input_length=maxlen
))

model.add(Flatten())

model.add(Dense(units=1,
                activation='sigmoid'))

model.compile(
    optimizer='rmsprop'
    , loss=keras.losses.binary_crossentropy
    , metrics=['acc']
)
model.summary()

# model.fit(x_train, y_train,
#           epochs=10,
#           validation_split=0.2)

# embding层可以使用预训练分词，比如 GloVe
