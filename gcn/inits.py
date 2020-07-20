import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""

    """
    均匀分布初始化方法，又称Xavier均匀初始化，
    参数从 [-limit, limit] 的均匀分布产生，
    其中limit为 sqrt(6 / (fan_in + fan_out))。
    fan_in为权值张量的输入单元数，
    fan_out是权重张量的输出单元数。
    该函数返回 [fan_in, fan_out]大小的Variable。
    """

    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)