import tensorflow as tf


def test_reduce_mean():
    """
    tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    第一个参数input_tensor： 输入的待降维的tensor;
    第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
    第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
    第四个参数name： 操作的名称;
    """
    a = [[1, 2, 3], [4, 5, 6]]
    b = tf.cast(a, tf.dtypes.float32)
    print(tf.reduce_mean(b, axis=1, keepdims=True).numpy())


def test_reduce_sum():
    """
    在不同的维度进行求和运算
    """
    a = [[1, 2, 3], [4, 5, 6]]
    print(tf.reduce_sum(a, axis=1).numpy())


def test_matrix():
    # 计算向量矩阵的值
    x = tf.constant([[1, 2], [3, 4]])
    y = tf.Variable([[0, 0], [0, 0]], shape=(2, 2))
    y.assign(tf.Variable([[3, 4], [5, 6]]))
    a = tf.add(x, y)
    print(a.numpy())

    # 循环计算
    x = tf.Variable(0)
    for _ in range(5):
        x.assign(tf.add(x, 1))
        print(x.numpy())
