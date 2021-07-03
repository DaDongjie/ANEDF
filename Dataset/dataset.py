import linecache
from Utils.utils import *
from scipy.sparse import *
from sklearn import metrics
import networkx as nx
import igraph as ig
import math
from networkx.algorithms import community
from sklearn.metrics.pairwise import cosine_similarity


class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.feature_file = config['feature_file']
        self.label_file = config['label_file']
        self.walks_file = config['walks_file']
        # W 图、X walks、 Z 特征、Y 标签
        self.W, self.X, self.Z, self.Y = self._load_data()

        # 余弦相似度
        attr = csr_matrix(self.Z).toarray()
        print("calculate cosine similarity..")
        flag = 0
        if (flag == 1):
            self.Z = metrics.pairwise.euclidean_distances(attr, attr)
        else:
            self.Z = metrics.pairwise.cosine_similarity(attr, attr)
        self.X_Z = np.hstack((self.X, self.Z))
        print(self.X_Z.shape)

        self.num_nodes = self.W.shape[0]
        self.num_feas = self.Z.shape[1]
        self.num_classes = self.Y.shape[1]
        self.num_edges = np.sum(self.W) / 2
        self.num_net_atts = self.X_Z.shape[1]
        print('nodes {}, edes {}, features {}, classes {}'.format(self.num_nodes, self.num_edges, self.num_feas, self.num_classes, self.num_net_atts))


        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False


    def _load_data(self):
        lines = linecache.getlines(self.label_file) # group.txt
        lines = [line.rstrip('\n') for line in lines]

        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0  # 统计标签数
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        num_classes = len(label_map)
        num_nodes = len(node_map)

        L = np.zeros((num_nodes, num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1

        #=========load feature==========
        lines = linecache.getlines(self.feature_file) # features.txt
        lines = [line.rstrip('\n') for line in lines]

        num_features = len(lines[0].split(' ')) - 1 # 特征数
        Z = np.zeros((num_nodes, num_features), dtype=np.float32)

        for line in lines:
            line = line.split(' ')
            node_id = node_map[line[0]]
            Z[node_id] = np.array([float(x) for x in line[1:]])



        #==========load graph========
        W = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.graph_file) # edges.txt
        lines = [line.rstrip('\n') for line in lines]
        i=0
        for line in lines:
            # i = i+1
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            W[idx2, idx1] = 1.0
            W[idx1, idx2] = 1.0
        # print("i=",i)

        #=========load walks========
        X = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.walks_file) # walks.txt
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            X[idx2, idx1] = 1.0
            X[idx1, idx2] = 1.0


        return W, X, Z, L


    def sample(self, batch_size, do_shuffle=True, with_label=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self._order) # shuffle（x） x 打乱的数组或列表
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()

        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]
        mini_batch.X = self.X[cur_index]
        mini_batch.adj = self.W[cur_index][:, cur_index]
        mini_batch.Z = self.Z[cur_index]
        mini_batch.X_Z = self.X_Z[cur_index]
        if with_label:
            mini_batch.Y = self.Y[cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True
        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):
        mini_batch = Dotdict()
        mini_batch.X = self.X[idx]
        mini_batch.Z = self.Z[idx]
        mini_batch.W = self.W[idx][:, idx]
        mini_batch.X_Z = self.X_Z[idx]
        # W 图、X walks、 Z 特征、Y 标签

        return mini_batch

