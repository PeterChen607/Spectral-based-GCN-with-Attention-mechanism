import tensorflow as tf
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh


indices = np.array([[0, 1],
                    [0, 3],], dtype=np.int64)
values = np.array([2, 1], dtype=np.int64)
dense_shape = np.array([4, 4], dtype=np.int64)

x = tf.SparseTensor(indices, values, dense_shape)
print(x)

x = tf.sparse_add(x, tf.sparse_transpose(x, [1, 0]))/2

x = tf.sparse_tensor_to_dense(x)
rowsum = tf.reduce_sum(x, 1)
# d_inv_sqrt = np.power(rowsum, -0.5).flatten()       #  D^(-1/2)
d_inv_sqrt = tf.diag(tf.pow(rowsum, 0)) 
d_inv_sqrt = tf.multiply(d_inv_sqrt, 2)
largest_eigval, _ = tf.self_adjoint_eig(d_inv_sqrt)  # \lambda_{max}



with tf.Session() as sess:
    print(x.eval())
    print("rowsum:", rowsum.eval())
    print("d", d_inv_sqrt.eval())
    print('egi', largest_eigval.eval())

# with tf.Session() as sess:
#     # 这么写就是为了打印值
#     sparse_tensor = sess.run(x, feed_dict={
#         x: tf.SparseTensorValue(indices, values, dense_shape)})
#     print('tensor', sparse_tensor)
#     tensor_value = tf.sparse_tensor_to_dense(sparse_tensor)
#     print('tensor表示的稀疏矩阵:\n', sess.run(tensor_value))

#     sparse_tensor = sess.run(x, feed_dict={
#         x: tf.SparseTensorValue(indices, values, dense_shape)})
#     sparse_tensor = tf.sparse_transpose(sparse_tensor, [1, 0])
#     print('tensor', sparse_tensor)
#     tensor_value = tf.sparse_tensor_to_dense(sparse_tensor)
#     print('tensor表示的稀疏矩阵:\n', sess.run(tensor_value))


