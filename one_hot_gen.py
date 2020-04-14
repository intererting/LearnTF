import numpy as np

# a = dict(zip((1, 2), ('a', 'b')))
# print(a.get('a'))

from tensorflow.keras.preprocessing.text import Tokenizer

#
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# # 散列空间越大,散列冲突几率越小
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
# sequences = tokenizer.texts_to_sequences(samples)
# print(sequences)
# one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index
print(word_index)
# print('Found %s unique tokens.' % len(word_index))
#
# print(one_hot_results)

# 散列的简单实现
# samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# dimensionality = 1000
# max_length = 10
# results = np.zeros((len(samples), max_length, dimensionality))
# for i, sample in enumerate(samples):
#     for j, word in list(enumerate(sample.split()))[:max_length]:
#         index = abs(hash(word)) % dimensionality
#         results[i, j, index] = 1.
