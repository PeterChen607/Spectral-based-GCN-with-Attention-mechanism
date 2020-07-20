from inits import *
import tensorflow as tf
from utils import *
import scipy.sparse as sp


flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


#稀疏矩阵的dropout操作
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


#定义Layer 层，主要作用是：对每层的name做了命名，还用一个参数决定是否做log
class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    #__call__ 的作用让 Layer 的实例成为可调用对象；
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

#根据 Layer 继承得到denseNet
class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act   #激活函数
        self.sparse_inputs = sparse_inputs  #是否是稀疏数据
        self.featureless = featureless  #输入的数据带不带特征矩阵
        self.bias = bias  #是否有偏置

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
        # https://blog.csdn.net/qq_41058594/article/details/85165025 对variable_scope的解释
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    #重写了_call 函数，其中对稀疏矩阵做 drop_out:sparse_dropout()
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


#从 Layer 继承下来得到图卷积网络，与denseNet的唯一差别是_call函数和__init__函数（self.support = placeholders['support']的初始化）
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.adj = placeholders['adjacency']
        self.output_dim = output_dim
        self.input_dim = input_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        # 下面是定义变量，主要是通过调用utils.py中的glorot函数实现
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            seq = tf.sparse_tensor_to_dense(x, validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
            seq = x
        

        seq = seq[np.newaxis]
        nb_nodes = 2708
        coef_drop = 0.1
        out_sz = self.output_dim
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
        adj_mat = self.adj[0]
        print('adj_mat', adj_mat)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        # 非对称矩阵转换为对称矩阵
        coefs = tf.sparse_add(coefs, tf.sparse_transpose(coefs))/2

        # print(coefs)

        coefs_mat = tf.sparse_tensor_to_dense(coefs, validate_indices=False)
        coefs_I = tf.diag(tf.pow(tf.reduce_sum(coefs_mat, 1), 0)) 
        coefs_mat = tf.add(coefs_mat, coefs_I)
        rowsum = tf.reduce_sum(coefs_mat, 1, keepdims=True)
        # print('rowsum:', rowsum)
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()       #  D^(-1/2)
        d_inv_sqrt = tf.pow(rowsum, -0.5)

        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_inv_inverse = tf.pow(d_inv_sqrt, -1)
        inf = tf.math.is_inf(d_inv_sqrt)
        inf_int = tf.to_float(inf)
        d_inv_inverse = tf.to_float(d_inv_inverse)
        d = tf.add(d_inv_inverse, inf_int)
        d = tf.pow(d, -1)
        d = tf.subtract(d, inf_int)

        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        d = tf.reduce_sum(d, 1)
        d_mat_inv_sqrt = tf.diag(d)

        # Support_mat =  coefs.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()      #transpose:转置
        Support_mat = tf.multiply(tf.transpose(tf.multiply(coefs_mat, d_mat_inv_sqrt)), d_mat_inv_sqrt)
        print('support_mat:', Support_mat)

        # convolve
        # convolve 卷积的实现。主要是根据论文中公式Z = \tilde{D}^{-1/2}\tilde{A}^{-1/2}X\theta实现
        supports = list()  #support是邻接矩阵的一个变化
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            # support = dot(self.support[i], pre_sup, sparse=True)
            support = dot(Support_mat, pre_sup, sparse=False)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)