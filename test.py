import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# print(tf.test.is_gpu_available())

#
# a = np.array([[[1, 3]], [[2, 4]]])
# print(np.mean(a, axis=(1, 2))[0])

# 矩阵与向量的乘法，位置不一样结果不一样
# a = np.array([[1, 2], [3, 4], [5, 6]])
# b = np.array([2, 2])
# #
# print(a.shape)
# print(b.shape)
# print(np.dot(a, b))
# print(np.dot(b, a))

# print(np.random.randint(1, 10, size=10))

# print(np.arange(1, 10))

# print(np.array([[1, 2, 3], [4, 5, 6]])[:, ::-1])

# a = np.array((None,))
# print(a.shape)
# n : 试验次数
# pvals : p长度序列，表示每次的概率
# size : int or tuple of ints, optional
# 例子：
# 1.投一个均匀骰子20次：
#
# np.random.multinomial(20, [1/6.]*6, size=1)
# result = np.random.multinomial(1, [1 / 6.] * 6, size=1)
# print(result,type( np.argmax(
#     result
# )))
# [[10  1  1  4  3  1]]
# 表示出现每个面的次数

# def sample(preds, temperature=1.0) -> [int]:
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# print('abcd'[sample([1. / 3, 1. / 3, 1. / 3], temperature=0.5)])

a = [[1, 1], [20, 20]]
print(K.prod(K.cast(K.shape(a), 'float32')))
