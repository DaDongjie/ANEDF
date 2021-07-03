import networkx as nx
import numpy as np
import linecache
import random
from scipy.sparse import dok_matrix



class Graph(object):
    def __init__(self, config):
        self.G = None
        self.is_adjlist = config['is_adjlist'] # False
        self.graph_file = config['graph_file'] # './cora/edges.txt'  有向边
        self.label_file = config['label_file'] # './cora/labels.txt'  是group.txt
        self.feature_file = config['feature_file'] # './cora/features.txt'
        self.node_status_file = config['node_status_file'] # ''

        if self.is_adjlist:
            self.read_adjlist()
        else:
            self.read_edgelist()


        if self.label_file:
            self.read_node_label()

        if self.feature_file: # 应该是one-hot 向量
            self.read_node_features()

        if self.node_status_file:
            self.read_node_status()


        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()

        print('num of nodes: {}'.format(self.num_nodes))
        print('num of edges: {}'.format(self.num_edges))


    def encode_node(self):
        for id, node in enumerate(self.G.nodes()): # nodes（）返回一个列表 所有节点的名称，作为str  (n, data)
            # print(id, node) # id代表节点个数（索引），node 是节点，一共是2708个节点
            self.G.nodes[node]['id'] = id
            self.G.nodes[node]['status'] = ''  # 表示什么



    def read_adjlist(self):
        self.G = nx.read_adjlist(self.graph_file, create_using=nx.DiGraph()) # G NetworkX graph，邻接表格式中与直线对应的图
        for i, j in self.G.edges(): # i,j是边的起点和终点
            self.G[i][j]['weight'] = 1.0

        self.encode_node()

    def read_edgelist(self):
        self.G = nx.DiGraph() # 创建一个没有节点和边的空图结构(“空图”) 有向

        lines = linecache.getlines(self.graph_file) # 从缓存中获取Python源文件的行
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')

            src = line[0]
            dst = line[1]

            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)

            # 权重赋值
            weight = 1.0
            if len(line) == 3:
                weight = float(line[2])
            self.G[src][dst]['weight'] = float(weight)
            self.G[dst][src]['weight'] = float(weight) #无向边，权重为1

        self.encode_node()

    def read_node_label(self):
        lines = linecache.getlines(self.label_file) # group.txt
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['label'] = line[1:] # 标签赋值保存

    def read_node_features(self):
        lines = linecache.getlines(self.feature_file) # one-hot向量
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['feature'] = np.array([float(x) for x in line[1:]])
            # G.nodes[line[0]] 是特征行标，上面一行代码就是将特征保存在对一个的节点中

    def read_node_status(self):
        lines = linecache.getlines(self.feature_file)
        lines = [line.rstrip('\n') for line in lines]

        for line in lines:
            line = line.split(' ')
            self.G.nodes[line[0]]['status'] = line[1] # train test valid


class DeepWalker:
    def __init__(self, G):
        self.G = G.G


    def deepwalk_walk(self, walk_length, start_node): # 80
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur)) # 邻居节点
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))  # 从邻居节点随机选择1个节点加入漫步中
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1) + '/' + str(num_walks))
            random.shuffle(nodes) # shuffle直接在原来的数组上进行操作，改变原来数组的顺序
            for node in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=node)) # 添加一个walk[]  80步
        return walks


def get_walks(graph, config):
    num_walks = config['num_walks'] # 10
    walk_length = config['walk_length'] # 80
    window_size = config['window_size'] # 10
    walks_file = config['walks_file'] # './cora/walks.txt'

    walker = DeepWalker(graph)
    walks = walker.simulate_walks(num_walks, walk_length) # （10 80） 返回模拟产生的随机游走

    num_nodes = graph.num_nodes
    adj_matrix = dok_matrix((num_nodes, num_nodes), np.float32) # dok_matrix((M,N), [dtype])
    # create the matrix with initial shape (M,N) 创建一个形状是num_nodes*num_nodes 的矩阵 ，类型是float32

    node_map = {} # id->node
    for node in graph.G.nodes():
        node_map[graph.G.nodes[node]['id']] = node  # graph.G.nodes[node]['id']=id

    # print(type(walks))
    # print(len(walks)) # 游走 2708个节点，10次迭代，每次游走长度是80

    for line in walks: # walks 是二维列表 长度是27080 每个line长度是80
        # print(line)
        for pos, node in enumerate(line): # pos是node的索引位置 0-79
            start = max(0, pos - window_size) # window_size=10
            # 在enumerate()函数中，也可以给第2个参数“start”的值，指定当前索引的初始值
            for pos2, node2 in enumerate(line[start:(pos + window_size + 1)], start):
                if pos2 != pos:
                    src = graph.G.nodes[node]['id'] # node->id
                    dst = graph.G.nodes[node2]['id']
                    adj_matrix[src, dst] = 1.0
                    adj_matrix[dst, src] = 1.0

    edge_list = []
    print("==================")
    for item in adj_matrix.items():
        # print("123:",item) # item:((2127, 1661), 1.0)
        src = item[0][0]
        dst = item[0][1]
        if dst > src:
            edge_list.append(node_map[src] + ' ' + node_map[dst])  #id->node

    with open(walks_file, 'w') as fid:
        fid.write('\n'.join(edge_list))





