import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import pdb
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(' ')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def check_reconstruction(embedding, graph_data, check_index):
    def get_precisionK(embedding, data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / data.N
            y = ind % data.N
            count += 1
            if (data.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret

def check_link_prediction(embedding, train_graph_data, origin_graph_data, check_index):
    def get_precisionK(embedding, train_graph_data, origin_graph_data, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        N = train_graph_data.N
        for ind in sortedInd:
            x = ind / N
            y = ind % N
            if (x == y or train_graph_data.adj_matrix[x].toarray()[0][y] == 1):
                continue 
            count += 1
            if (origin_graph_data.adj_matrix[x].toarray()[0][y] == 1):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
    precisionK = get_precisionK(embedding, train_graph_data, origin_graph_data, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return ret
 
'''
def check_multi_label_classification(X, Y, test_ratio = 0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    return "micro_f1: %.4f macro_f1 : %.4f" % (micro, macro)
    #############################################
'''

def small_trick(y_test, y_pred):
    y_pred_new = np.zeros(y_pred.shape, np.bool)
    sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = sum(y_test[i])
        for j in range(num):
            y_pred_new[i][sort_index[i][j]] = True
    return y_pred_new

from sklearn.svm import LinearSVC
def multi_label_classification(X, Y, ratio):
    X = preprocessing.normalize(X, norm='l2')
    print("@@@@",type(X))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    logreg = LogisticRegression()
    svc = LinearSVC()
    c = 2.0 ** np.arange(-10, 10)

    # =========train=========
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=1)  #
    clf.fit(X_train, y_train)
    print('Best parameters')
    print(clf.best_params_)

    # =========test=========

    y_pred = clf.predict_proba(X_test)
    print("111:", type(y_test))
    print("111:", type(y_pred))
    y_pred = small_trick(y_test, y_pred)
    print("112:", type(y_test))
    print("112:", type(y_pred))
    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    # acc = accuracy_score(y_test, y_pred)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=2018)
    # clf = LinearSVC()
    # clf.fit(X_train, y_train)
    #
    # y_pred = clf.predict(X_test)
    #
    # macro = f1_score(y_test, y_pred, average='macro')
    # micro = f1_score(y_test, y_pred, average='micro')

    # fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)

    # AUC_s=auc(fpr, tpr, reorder=False)
    AUC_s = 0
    # print("acc: %.4f" % (acc))
    # print("AUC: %.4f" % (AUC_s))
    print("micro_f1: %.4f" % (micro))
    print("macro_f1: %.4f" % (macro))

    return micro, macro


def acc_val(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64) # 转换数据类型 转成int64
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def node_clustering(emb, one_hots):
    label = [np.argmax(one_hot) for one_hot in one_hots] # 由one-hot转换为普通np数组
    ClusterNUm = np.unique(label)


    clf = KMeans(n_clusters=len(ClusterNUm),init="k-means++")
    kmeans = clf.fit(emb)  # 学习到的嵌入表示

    cluster_groups = kmeans.labels_
    acc =acc_val(np.array(label),np.array(cluster_groups))
    nmi = metrics.normalized_mutual_info_score(label,cluster_groups)
    return acc,nmi

def write_embedding(embedding_result, outputFileName):
    f = open(outputFileName, 'w')
    N, dims = embedding_result.shape

    for i in range(N):
        s = ''
        for j in range(dims):
            if j == 0:
                s = str(i) + ',' + str(embedding_result[i, j])
            else:
                s = s + ',' + str(embedding_result[i, j])
        f.writelines(s + '\n')
    f.close()

def check_multi_label_classification(X, Y, ratio):

    X = preprocessing.normalize(X, norm='l2')

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)

    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)
    y_pred = small_trick(y_test, y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return micro, macro


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def swish(x, b = 1):
    return x * tf.nn.sigmoid(b * x)

def jaccard(edge_path):
    neighbor_set_dict={}
    node_set=set()  # 节点集合
    edge_num=0
    with open(edge_path) as fp:
        for line in fp.readlines():
            edge_num+=1
            node1=int(line.strip('\n\r').split()[0])
            node2=int(line.strip('\n\r').split()[1])
            node_set.add(node1)
            node_set.add(node2)
            if node1 not in neighbor_set_dict:  # 嵌套字典{node1：{ 邻居节点},}
                neighbor_set_dict[node1]=set()
                neighbor_set_dict[node1].add(node2)
            else:
                neighbor_set_dict[node1].add(node2)
            if node2 not in neighbor_set_dict:
                neighbor_set_dict[node2]=set()
                neighbor_set_dict[node2].add(node1)
            else:
                neighbor_set_dict[node2].add(node1)


    node_num=len(node_set)
    print('Node number:',node_num)
    print('Edge number:',edge_num)
    num=0
    sim_mat=np.zeros((node_num,node_num))  # 相似度矩阵
    row=[]
    col=[]
    data=[]
    for i in range(node_num):
        for j in range(node_num):
            i_nbr=neighbor_set_dict[i]  # <class 'set'>
            # print("i的邻居：",i,j, i_nbr)
            j_nbr=neighbor_set_dict[j]
            inter=len(i_nbr.intersection(j_nbr))  # 找出集合中相同的元素 个数
            union=len(i_nbr.union(j_nbr)) # 找出集合的并集 个数
            # print(inter,union)
            score=float(inter)/union   # Jaccard系数
            sim_mat[i,j]=score
            if i!=j and score>0:
                num+=1
                row.append(i)
                col.append(j)
                data.append(score)
    M=csr_matrix((data, (row, col)), shape=(node_num, node_num))  # 压缩稀疏矩阵
    # csr_matrix((data, (row_ind, col_ind)), [shape = (M, N)])
    # where ``data``, ``row_ind`` and ``col_ind`` satisfy the relationship ``data[row_ind[k], col_ind[k]] = data[k] ``.

    # print("1:", type(M))
    # print(M.shape)
    #np.savetxt('./data/cora_5429/sim_mat.txt', sim_mat, delimiter=' ', fmt='%s')


# 自己加的高阶
#     for i in range(node_num):
#         for j in range(i + 1, node_num):
#
#             if sim_mat[i][j] != 0.0:
#                 temp_i = i
#                 temp_j = j
#                 # print(sim_mat[i][j])
#                 # print("11:", temp_i, temp_j)
#                 for temp_i in range(temp_i + 1, node_num):
#                     if sim_mat[temp_i][j] != 0 and j < node_num-1:
#                         # print("12:", temp_i, temp_j)
#                         if temp_i != j:
#                             sim_mat[i][temp_i] += sim_mat[i][j] * sim_mat[temp_i][j]
#                             sim_mat[temp_i][i] = sim_mat[i][temp_i]
#
#     print("================")
#     np.savetxt('./data/cora_5429/sim_mat2.txt', sim_mat, delimiter=' ', fmt='%s')
#     # np.savetxt('./data/cora_5429/csr_matrix.txt', M, delimiter=' ', fmt='%s')
#     print('Jaccard simiarity finished!')
#     print(float(num)/(node_num*node_num))

    return M.toarray(),node_num

def feature_sim_mat(feature_path): # 计算属性 余弦相似度矩阵

    with open(feature_path) as fp:
        lines = fp.readlines()
        node_num = len(lines)
        line = lines[0]
        attr_num = len(line.strip('\n\r').split())
    print('Node number:', node_num, 'Attribute dimension:', attr_num)

    A = np.zeros((node_num, attr_num))  # 定义属性矩阵 2708*1434
    with open(feature_path) as fp:
        line_num = 0
        for line in fp.readlines():
            A[line_num, :] = line.strip('\n\r').split()
            line_num += 1

    print("A.shape:", A.shape)
    A_sim = cosine_similarity(A)  # 属性 余弦相似度矩阵    ##########################################################
    print("A_sim.shape:", A_sim.shape)
    print("A_sim_avg:", np.mean(A_sim))
    return A_sim

def PPMI(edge_path,window_size):
    G_dic = {}
    max_node = 0
    with open(edge_path) as f:
        lines = f.readlines()
        print('Edge Number:',len(lines))
        for line in lines:
            items = line.strip('\n').split()
            a = int(items[0])
            b = int(items[1])
            #if a == b:
            #    continue
            max_node = max(max_node, a)
            max_node = max(max_node, b)
            if a in G_dic:
                G_dic[a].append(b)
            else:
                G_dic[a] = [b]
            if b in G_dic:
                G_dic[b].append(a)
            else:
                G_dic[b] = [a]
    G = [[] for _ in range(max_node + 1)]
    for k, v in G_dic.items():
        G[k] = v
    node_num=len(G_dic.items())
    print('Node num:',node_num)
    walk_length=80
    walk_num=20
    walks = []
    for cnt in range(walk_num):
        for node in range(node_num):
            path = [node]
            while len(path) < walk_length:
                cur = path[-1]
                if len(G[cur]) > 0:
                    path.append(random.choice(G[cur]))
                else:
                    break
            walks.append(path)

    vocab = np.zeros(node_num)
    for walk in walks:
        for node in walk:
            vocab[node] += 1
    pair_num_dict={}
    for walk in walks:
        for i in range(len(walk)):
            source_node = walk[i]
            left_window = max(i - window_size, 0)
            right_window = min(i + window_size, len(walk))
            for j in range(left_window, right_window):
                target_node=walk[j]
                if source_node!=target_node:
                    if (source_node,target_node) not in pair_num_dict:
                        pair_num_dict[(source_node,target_node)]=1
                    else:
                        pair_num_dict[(source_node, target_node)] += 1
    PPMI_matrix=np.zeros((node_num,node_num))
    len_D=node_num*walk_length*walk_num
    for key in pair_num_dict:
        node1=key[0]
        node2=key[1]
        co_occurance=pair_num_dict[key]
        frequency_1=vocab[node1]
        frequency_2=vocab[node2]
        res=np.log(1.0*co_occurance*len_D/(frequency_1*frequency_2))
        if res<0:
            res=0
        PPMI_matrix[node1,node2]=res
    return PPMI_matrix,node_num

