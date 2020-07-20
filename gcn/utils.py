import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from csv_output import csv_output


# def extend_mask_and_label(graph, labels):


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    #     print(int(line.strip()))
    # print("min", min(index))
    return index


def sample_mask(idx, l):
    """Create mask."""
    for i in idx:
        mask = np.zeros(l)
        mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# 数据的读取，这个预处理是把训练集（其中一部分带有标签），测试集，标签的位置，对应的掩码训练标签等返回。
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):  # get python version
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    # x.shape:(140, 1433); y.shape:(140, 7);tx.shape:(1000, 1433);ty.shape:(1708, 1433);
    # allx.shape:(1708, 1433);ally.shape:(1708, 7)
    x, y, tx, ty, allx, ally, graph = tuple(objects)  # 转化成tuple

    # 测试数据集
    # print(x[0][0],x.shape,type(x))  ##x是一个稀疏矩阵,记住1的位置,140个实例,每个实例的特征向量维度是1433  (140,1433)
    # print(y[0],y.shape)   ##y是标签向量,7分类，140个实例 (140,7)

    ##训练数据集
    # print(tx[0][0],tx.shape,type(tx))  ##tx是一个稀疏矩阵,1000个实例,每个实例的特征向量维度是1433  (1000,1433)
    # print(ty[0],ty.shape)   ##y是标签向量,7分类，1000个实例 (1000,7)

    ##allx,ally和上面的形式一致
    # print(allx[0][0],allx.shape,type(allx))  ##tx是一个稀疏矩阵,1708个实例,每个实例的特征向量维度是1433  (1708,1433)
    # print(ally[0],ally.shape)   ##y是标签向量,7分类，1708个实例 (1708,7)

    #graph是一个字典，大图总共2708个节点
    count = 0
    for i in range(0, 2708):
        for j in graph[i]:
            count +=1
    print("edge count:", count)


    # 测试数据集的索引乱序版
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # print(test_idx_reorder)
    # [2488, 2644, 3261, 2804, 3176, 2432, 3310, 2410, 2812,...]

    # 从小到大排序,如[1707,1708,1709,...]
    test_idx_range = np.sort(test_idx_reorder)

    # 处理citeseer中一些孤立的点
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position

        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        # print("test_idx_range_full.length",len(test_idx_range_full))
        # test_idx_range_full.length 1015

        # 转化成LIL格式的稀疏矩阵,tx_extended.shape=(1015,1433)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        # print(tx_extended)
        # [2312 2313 2314 2315 2316 2317 2318 2319 2320 2321 2322 2323 2324 2325
        # ....
        # 3321 3322 3323 3324 3325 3326]

        # test_idx_range-min(test_idx_range):列表中每个元素都减去min(test_idx_range)，即将test_idx_range列表中的index值变为从0开始编号
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        # print(tx_extended.shape) #(1015, 3703)

        # print(tx_extended)
        # (0, 19) 1.0
        # (0, 21) 1.0
        # (0, 169) 1.0
        # (0, 170) 1.0
        # (0, 425) 1.0
        #  ...
        # (1014, 3243) 1.0
        # (1014, 3351) 1.0
        # (1014, 3472) 1.0

        tx = tx_extended
        # print(tx.shape)
        # (1015, 3703)
        # 997,994,993,980,938...等15行全为0

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        # for i in range(ty.shape[0]):
        #     print(i," ",ty[i])
        #     # 980 [0. 0. 0. 0. 0. 0.]
        #     # 994 [0. 0. 0. 0. 0. 0.]
        #     # 993 [0. 0. 0. 0. 0. 0.]

    # 将allx和tx叠起来并转化成LIL格式的feature,即输入一张整图
    features = sp.vstack((allx, tx)).tolil()
    
    # 把特征矩阵还原，和对应的邻接矩阵对应起来，因为之前是打乱的，不对齐的话，特征就和对应的节点搞错了。
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # csv_output("features.csv", features)
    # for i in range(40):
    #     print(features[i])
    print("features.shape:",features.shape)
    # features.shape: (2708, 1433)

    # 邻接矩阵格式也是LIL的，并且shape为(2708, 2708), 邻接矩阵是全连接，实例数大小的方阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # for i in graph[1]:
    #     print(i, adj[i])


    # labels.shape:(2708, 7)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # for i in range(700, 1000):
    #     print(labels[i])
    # csv_output("labels.csv", labels)

    # for i in range(140):
    #     for j in graph[i]:


    # len(list(idx_val)) + len(list(idx_train)) + len(idx_test) =  1640
    idx_test = test_idx_range.tolist()
    # print(idx_test)
    # [1708, 1709, 1710, 1711, 1712, 1713,...,2705, 2706, 2707]
    # print(len(idx_test))
    # 1000

    idx_train = range(len(y))
    # idx_train = range(9000)
    print("idx_train: ", idx_train, len(idx_train))
    # range(0, 140)

    idx_val = range(len(y), len(y) + 500)
    # idx_val = range(9000, 9000 + 5000)
    print("idx_val: ", idx_val,len(idx_val))
    # range(140, 640) 500

    # 训练mask：idx_train=[0,140)范围的是True，后面的是False
    train_mask = sample_mask(idx_train, labels.shape[0])  # labels.shape[0]:(2708,)
    # print(train_mask,train_mask.shape)
    # [True  True  True... False False False]  # labels.shape[0]:(2708,)

    # 验证mask：val_mask的idx_val=(140, 640]范围为True，其余的是False
    val_mask = sample_mask(idx_val, labels.shape[0])  # labels.shape[0]:(2708,)

    # test_mask，idx_test=[1708,2707]范围是True，其余的是False
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    print(y_train.shape," ",y_test.shape," ",y_val.shape)
    # (2708, 7)(2708, 7)(2708, 7)

    # 替换了true位置
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    # print(y_val[610:710])
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


# 将稀疏矩sparse_mx阵转换成tuple格式并返回
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# 处理特征:特征矩阵进行归一化并返回一个格式为(coords, values, shape)的元组
# 特征矩阵的每一行的每个元素除以行和，处理后的每一行元素之和为1
# 处理特征矩阵，跟谱图卷积的理论有关，目的是要把周围节点的特征和自身节点的特征都捕捉到，同时避免不同节点间度的不均衡带来的问题
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    print("preprocess_features")
    # >> > b = [[1.0, 3], [2, 4], [3, 5]]
    # >> > b = np.array(b)
    # >> > b
    # array([[1., 3.],
    #        [2., 4.],
    #        [3., 5.]])
    # >> > np.array(b.sum(1))
    # array([4., 6., 8.])
    # >> > c = np.array(b.sum(1))
    # >> > np.power(c, -1)
    # array([0.25, 0.16666667, 0.125])
    # >> > np.power(c, -1).flatten()
    # array([0.25, 0.16666667, 0.125])
    # >> > r_inv = np.power(c, -1).flatten()
    # >> > import scipy.sparse as sp
    # >> > r_mat_inv = sp.diags(r_inv)
    # >> > r_mat_inv
    # < 3x3 sparse matrix of type '<class 'numpy.float64 '>'
    # with 3 stored elements (1 diagonals) in DIAgonal format >
    # >> > r_mat_inv.toarray()
    # array([[0.25, 0., 0.],
    #        [0., 0.16666667, 0.],
    #        [0., 0., 0.125]])
    # >> > f = r_mat_inv.dot(b)
    # >> > f
    # array([[0.25, 0.75],
    #        [0.33333333, 0.66666667],
    #        [0.375, 0.625]])

    # a.sum()是将矩阵中所有的元素进行求和;a.sum(axis = 0)是每一列列相加;a.sum(axis = 1)是每一行相加
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    # print("r_inv:", r_inv)
    # r_inv: [0.11111111 0.04347826 0.05263158... 0.05555556 0.07142857 0.07692308]
    # np.isnan(ndarray)返回一个判断是否是NaN的bool型数组
    r_inv[np.isinf(r_inv)] = 0.
    # sp.diags创建一个对角稀疏矩阵
    r_mat_inv = sp.diags(r_inv)
    # dot矩阵乘法
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


# 邻接矩阵adj对称归一化并返回coo存储模式
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()       #  D^(-1/2)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()      #transpose:转置


# 将邻接矩阵加上自环以后，对称归一化，并存储为COO模式，最后返回元组格式
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    #加上自环，再对称归一化
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


# 构建输入字典并返回
#labels和labels_mask传入的是具体的值，例如
# labels=y_train,labels_mask=train_mask；
# labels=y_val,labels_mask=val_mask；
# labels=y_test,labels_mask=test_mask；
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    #由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # print(features)
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
    #49126是特征矩阵存储为coo模式后非零元素的个数（2078*1433里只有49126个非零，稀疏度达1.3%）
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# 切比雪夫多项式近似:计算K阶的切比雪夫近似矩阵
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)  # D^{-1/2}AD^{1/2}
    laplacian = sp.eye(adj.shape[0]) - adj_normalized  # L = I_N - D^{-1/2}AD^{1/2}
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')  # \lambda_{max}
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  # 2/\lambda_{max}L-I_N

    # 将切比雪夫多项式的 T_0(x) = 1和 T_1(x) = x 项加入到t_k中
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    # 依据公式 T_n(x) = 2xT_n(x) - T_{n-1}(x) 构造递归程序，计算T_2 -> T_k
    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

# load_data('cora')

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape