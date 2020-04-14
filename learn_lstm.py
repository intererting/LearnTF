import numpy as np
from tensorflow import keras

np.random.seed(0)  # Set a random seed for reproducibility

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = keras.Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = keras.layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = keras.layers.LSTM(32)(x)

auxiliary_output = keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = keras.Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)

model = keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

headline_data = np.round(np.abs(np.random.rand(100, 100) * 100))
additional_data = np.random.randn(100, 5)
headline_labels = np.random.randn(100, 1)
additional_labels = np.random.randn(100, 1)
model.fit([headline_data, additional_data], [headline_labels, additional_labels],
          epochs=50, batch_size=32)

model.predict({'main_input': headline_data, 'aux_input': additional_data})
