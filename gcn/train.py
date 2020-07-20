from __future__ import division
#即使在python2.X，使用print就得像python3.X那样加括号使用。
from __future__ import print_function
# 导入python未来支持的语言特征division(精确除法)，
# 当我们没有在程序中导入该特征时，"/"操作符执行的是截断除法(Truncating Division)；
# 当我们导入精确除法之后，"/"执行的是精确除法, "//"执行截断除除法

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
#第一层的输出维度
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')

#权值衰减：防止过拟合
# loss计算方式（权值衰减+正则化）：self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') #K阶的切比雪夫近似矩阵的参数k



# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# print(features)
# (0, 19) 1.0
# (0, 81) 1.0
# ...
# (2707, 1412) 1.0
# (2707, 1414) 1.0

# print(type(features))
# <class 'scipy.sparse.lil.lil_matrix'>

#预处理特征矩阵:将特征矩阵进行归一化并返回tuple (coords, values, shape)
features = preprocess_features(features)
# for i in features:
#     print(features)
# (array([[   0, 1274],
#        [   0, 1247],
#        [   0, 1194],
#        ...,
#        [2707,  329],
#        [2707,  186],
#        [2707,   19]], dtype=int32), array([0.11111111, 0.11111111, 0.11111111, ..., 0.07692308, 0.07692308,
#        0.07692308], dtype=float32), (2708, 1433))

# print(type(features))
# <class 'tuple'>

# print("features[1]",features[1])
# features[1] [0.11111111 0.11111111 0.11111111 ... 0.07692308 0.07692308 0.07692308]

# print("features[1].shape",features[1].shape)
# features[1].shape (49216,)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]  #support是邻接矩阵的归一化形式
    # print("support：",support)
    # support： [(array([[0, 0],
    #                   [633, 0],
    #                   [1862, 0],
    #                   ...,
    #                   [1473, 2707],
    #                   [2706, 2707],
    #                   [2707, 2707]], dtype=int32), array([0.25, 0.25, 0.2236068, ..., 0.2, 0.2,
    #                                                       0.2]), (2708, 2708))]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))


adj1 = [preprocess_adj_bias(adj)]
# print('adj_tuple', adj1)


# print("num_supports:",num_supports)
#num_supports: 1

# Define placeholders
placeholders = {
    #由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # features也是稀疏矩阵，也用LIL格式表示，因此定义为tf.sparse_placeholder(tf.float32)，维度(2708, 1433)
    # print(features[2])
    # (2708, 1433)
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'adjacency': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # print(y_train.shape[1])
    # 7
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}




# Create model
# print(features[2][1])
# 1433
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# print("GCN output_dim:",model.output_dim)
#GCN output_dim: 7


# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['adjacency'][i]: adj1[i] for i in range(len(adj1))})
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    # print('feed_dict1')
    # for i in feed_dict:
    #     print(i)
    feed_dict.update({placeholders['adjacency'][i]: adj1[i] for i in range(len(adj1))})
    # print('feed_dict2')
    # for i in feed_dict:
    #     print(i)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # print('feed_dict3')
    # for i in feed_dict:
    #     print(i)

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # print("outs:",outs) #outs: [None, 0.57948196, 0.9642857]


    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    FLAGS.learning_rate = 0.85**epoch * FLAGS.learning_rate

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))