from Utils.utils import *
from sklearn.cluster import KMeans
from Utils import utils
from sklearn.metrics.pairwise import cosine_similarity


class Trainer(object):

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.net_att_input_dim = config['net_att_input_dim']
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.net_att_shape = config['net_att_shape']
        self.drop_prob = config['drop_prob']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        # self.sita = config['sita']

        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']

        # tf.placeholder(dtype,shape=None,name=None) 参数：数据类型，形状[任意，***]，name：名称
        # 函数用于定义过程，在执行的时候再赋具体的值
        self.x = tf.placeholder(tf.float32, [None, self.net_input_dim])
        self.z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.w = tf.placeholder(tf.float32, [None, None])   # 维度是2708*2708  不匹配 mini_batch100*100
        self.x_z = tf.placeholder(tf.float32, [None, self.net_att_input_dim])
        self.emd = tf.placeholder(tf.float32, [None, self.net_att_input_dim])


        # 负采样
        self.neg_x = tf.placeholder(tf.float32, [None, self.net_input_dim])
        self.neg_z = tf.placeholder(tf.float32, [None, self.att_input_dim])
        self.neg_w = tf.placeholder(tf.float32, [None, None])   # ##########
        self.neg_x_z = tf.placeholder(tf.float32, [None, self.net_att_input_dim])
        self.neg_emd = tf.placeholder(tf.float32, [None, self.net_att_input_dim])
        # self.x 和 self.neg_x是一样的吗

        self.optimizer, self.loss = self._build_training_graph()
        # self.net_H, self.att_H, self.H = self._build_eval_graph()
        self.net_H, self.att_H, self.H, self.net_att_H = self._build_eval_graph()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def _build_training_graph(self):

        net_H, net_recon = self.model.forward_net(self.x, drop_prob=self.drop_prob, reuse=False)  # 原始网络结构、重构网络结构
        neg_net_H, neg_net_recon = self.model.forward_net(self.neg_x, drop_prob=self.drop_prob, reuse=True)

        att_H, att_recon = self.model.forward_att(self.z, drop_prob=self.drop_prob, reuse=False)
        neg_att_H, neg_att_recon = self.model.forward_att(self.neg_z, drop_prob=self.drop_prob, reuse=True)

        net_att_H, net_att_recon = self.model.forward_net_att(self.x_z, drop_prob=self.drop_prob, reuse=False)
        neg_net_att_H, neg_net_att_recon = self.model.forward_net_att(self.neg_x_z, drop_prob=self.drop_prob, reuse=True)




        self.emb = tf.concat([tf.nn.l2_normalize(net_recon, dim=1), tf.nn.l2_normalize(att_recon, dim=1)], axis=1)
        self.neg_emb = tf.concat([tf.nn.l2_normalize(neg_net_recon, dim=1), tf.nn.l2_normalize(neg_att_recon, dim=1)], axis=1)


        #================high-order proximity & semantic proximity=============
        # tf.square(x, name=none) 对x内的所有元素进行平方操作
        # tf.reduce_sum(tf.square(x, name=none)， 1) 求和，第二维  1 表示按行求和
        recon_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - net_recon), 1)) # 网络结构重构的损失
        recon_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_x - neg_net_recon), 1)) # neg_x加入负采样的网络
        recon_loss_3 = tf.reduce_mean(tf.reduce_sum(tf.square(self.z - att_recon), 1))
        recon_loss_4 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_z - neg_att_recon), 1))
        recon_loss_5 = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_z - net_att_recon), 1))  # 网络结构重构的损失
        recon_loss_6 = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_x_z - neg_net_att_recon), 1))  # neg_x加入负采样的网络

        recon_loss_xz = tf.reduce_mean(tf.reduce_sum(tf.square(self.emb - net_att_recon), 1))
        recon_loss_neg_xz = tf.reduce_mean(tf.reduce_sum(tf.square(self.neg_emb - neg_net_att_recon), 1))

        recon_loss = recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4 + recon_loss_5 + recon_loss_6 + recon_loss_xz + recon_loss_neg_xz
        # recon_loss = recon_loss_1 + recon_loss_2 + recon_loss_3 + recon_loss_4
        # 考虑结构、属性权重影响
        # 二次凸优化 是针对矩阵的 如何应用？

        #===============cross modality proximity==================
        # tf.multiply(net_H, att_H) 两个矩阵中对应元素各自相乘

        # pre_logit_pos = tf.reduce_sum(tf.multiply(net_H, att_H), 1) # 元素级别的相乘，有人就是两个相乘的数元素各自相乘，而不是矩阵相乘，按照第二维的元素求和  行
        # pre_logit_neg_1 = tf.reduce_sum(tf.multiply(neg_net_H, att_H), 1)
        # pre_logit_neg_2 = tf.reduce_sum(tf.multiply(net_H, neg_att_H), 1)
        #
        # # 损失计算
        # pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pre_logit_pos), logits=pre_logit_pos)
        # neg_loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_1), logits=pre_logit_neg_1)
        # neg_loss_2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pre_logit_neg_2), logits=pre_logit_neg_2)
        #
        # cross_modal_loss = tf.reduce_mean(pos_loss + neg_loss_1 + neg_loss_2)


        #=============== first-order proximity================
        # tf.matmul(a, b, transpose_b=True) 将矩阵a乘以矩阵b transpose_b: 如果为真, b则在进行乘法计算前进行转置
        pre_logit_pp_x = tf.matmul(net_H, net_H, transpose_b=True)  # 结构的联合分布
        pre_logit_pp_z = tf.matmul(att_H, att_H, transpose_b=True)
        pre_logit_pp_x_z = tf.matmul(net_att_H, net_att_H, transpose_b=True)  # 负采样结构的联合分布
        pre_logit_nn_x = tf.matmul(neg_net_H, neg_net_H, transpose_b=True)  # 负采样结构的联合分布
        pre_logit_nn_z = tf.matmul(neg_att_H, neg_att_H, transpose_b=True)
        pre_logit_nn_x_z = tf.matmul(neg_net_att_H, neg_net_att_H, transpose_b=True)



        # 一阶相似性损失计算  # labels和logits的shape和type一样。

        # tf.nn.sigmoid_cross_entropy_with_logits()返回的是：一个张量，和logits的大小一致。
        # 返回的是逻辑损失：-labels * np.log(sigmoid(logits)) - (1 - labels) * np.log(1 - sigmoid(logits))
        # tf.diag_part()返回张量的对角元素，tf.ones_like返回形状一样，所有元素为1 的张量
        # labels = tf.ones_like(tf.diag_part(pre_logit_pp_x)) 返回[1 1 1 …… 1]

        pp_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.w)[0]),
                                                            logits=pre_logit_pp_x) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_pp_x)),
                                                              logits=tf.diag_part(pre_logit_pp_x))
        pp_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.w)[0]),
                                                            logits=pre_logit_pp_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_pp_z)),
                                                              logits=tf.diag_part(pre_logit_pp_z))
        pp_x_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.w)[0]),
                                                            logits=pre_logit_pp_x_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_pp_x_z)),
                                                              logits=tf.diag_part(pre_logit_pp_x_z))


        nn_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.neg_w)[0]),
                                                            logits=pre_logit_nn_x) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_nn_x)),
                                                              logits=tf.diag_part(pre_logit_nn_x))
        nn_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.neg_w)[0]),
                                                            logits=pre_logit_nn_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_nn_z)),
                                                              logits=tf.diag_part(pre_logit_nn_z))
        nn_x_z_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.w + tf.eye(tf.shape(self.neg_w)[0]),
                                                            logits=pre_logit_nn_x_z) \
                    - tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(tf.diag_part(pre_logit_nn_x_z)),
                                                              logits=tf.diag_part(pre_logit_nn_x_z))


        # first_order_loss = tf.reduce_mean(pp_x_loss + pp_z_loss + nn_x_loss + nn_z_loss)
        first_order_loss = tf.reduce_mean(pp_x_loss + pp_z_loss + pp_x_z_loss + nn_x_loss + nn_z_loss + nn_x_z_loss)

        # ========================权重分配=========================
        # ++++++++++++++++++++二次规划（凸优化）+++++++++++++++++++++
        # 分别在重构损失、一阶损失、高阶损失中应用权重分配

        # ========================损失计算=========================
        # loss = recon_loss * self.beta + first_order_loss * self.gamma + cross_modal_loss * self.alpha  # 原损失函数
        loss = recon_loss * self.beta + first_order_loss * self.gamma  # 原损失函数

        # tf.get_collection(key,scope=None)#返回：在由key指定的collection中的元素，经过scope的re.match筛选后的元素，组成的列表。
        # tf.GraphKeys.TRAINABLE_VARIABLES 可学习变量（一般指神经网络中的参数）
        vars_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net_encoder')
        vars_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'att_encoder')
        vars_net_att = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net_att_encoder')
        # print(vars_net)

        # 基于一定的学习率进行梯度优化训练  用于最小化loss，并更新var_list
        # opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_net+vars_att)
        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_net+vars_att+vars_net_att)

        return opt, loss

    def _build_eval_graph(self):

        net_H, _ = self.model.forward_net(self.x, drop_prob=0.0, reuse=True)
        att_H, _ = self.model.forward_att(self.z, drop_prob=0.0, reuse=True)
        net_att_H, _ = self.model.forward_net_att(self.x_z, drop_prob=0.0, reuse= True)
        # net_att_H, _ = self.model.forward_net_att(self.x_z, drop_prob=0.0, reuse= tf.AUTO_REUSE)

        # tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None) # x为输入的向量；dim为l2范化的维数，dim取值为0或1；epsilon的范化的最小值边界；
        # dim=1, 为按列进行l2范化 ;dim = 0, 为按行进行l2范化
        # tf.concat([],axis=1) # concat()是将tensor沿着指定维度连接起来, 对于2维来说，0表示行，1表示列

        # H = tf.concat([tf.nn.l2_normalize(net_H, dim=1), tf.nn.l2_normalize(att_H, dim=1)], axis=1)   # 连接网络结构和属性
        H = net_att_H   # 网络结构和属性
        print("111:", H.shape)
        print(net_att_H.shape)

        # return net_H, att_H, H
        return net_H, att_H, H, net_att_H


    def train(self, graph):

        for epoch in range(self.num_epochs):
            idx1, idx2 = self.generate_samples(graph)
            index = 0
            cost = 0.0
            cnt = 0
            while True:
                if index > graph.num_nodes: # graph.num_nodes:2708
                    break
                if index + self.batch_size < graph.num_nodes:  # batch_size =100
                    mini_batch1 = graph.sample_by_idx(idx1[index:index + self.batch_size])
                    mini_batch2 = graph.sample_by_idx(idx2[index:index + self.batch_size])
                else:
                    mini_batch1 = graph.sample_by_idx(idx1[index:])
                    mini_batch2 = graph.sample_by_idx(idx2[index:])

                index += self.batch_size

                loss, _ = self.sess.run([self.loss, self.optimizer],
                                        feed_dict={self.x: mini_batch1.X,
                                                   self.z: mini_batch1.Z,
                                                   self.x_z: mini_batch1.X_Z,
                                                   self.neg_x: mini_batch2.X,
                                                   self.neg_z: mini_batch2.Z,
                                                   self.neg_x_z: mini_batch2.X_Z,
                                                   self.w: mini_batch1.W,
                                                   self.neg_w: mini_batch2.W})

                # loss, _ = self.sess.run([self.loss, self.optimizer],
                #                         feed_dict={self.x: mini_batch1.X,
                #                                    self.z: mini_batch1.Z,
                #                                    self.neg_x: mini_batch2.X,
                #                                    self.neg_z: mini_batch2.Z,
                #                                    self.w: mini_batch1.W,
                #                                    self.neg_w: mini_batch2.W})


                cost += loss
                cnt += 1

                if graph.is_epoch_end:
                    break
            cost /= cnt   # 平均损失

            if epoch % 50 == 0:

                train_emb = None
                train_label = None
                while True:
                    mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True)

                    # emb = self.sess.run(self.H,
                    #                     feed_dict={self.x: mini_batch.X,
                    #                                self.z: mini_batch.Z})

                    emb = self.sess.run(self.H,
                                        feed_dict={self.x: mini_batch.X,
                                                   self.z: mini_batch.Z,
                                                   self.x_z: mini_batch.X_Z})

                    if train_emb is None:
                        train_emb = emb
                        train_label = mini_batch.Y
                    else:
                        train_emb = np.vstack((train_emb, emb))
                        train_label = np.vstack((train_label, mini_batch.Y))

                    if graph.is_epoch_end:
                        break
                micro_f1, macro_f1 = check_multi_label_classification(train_emb, train_label, 0.5)
                print('Epoch-{}, loss: {:.4f}, Micro_f1 {:.4f}, Macro_fa {:.4f}'.format(epoch, cost, micro_f1, macro_f1))

        self.save_model()


    def infer(self, graph):
        self.sess.run(tf.global_variables_initializer())
        self.restore_model()
        print("Model restored from file: %s" % self.model_path)

        train_emb = None
        train_label = None
        while True:
            mini_batch = graph.sample(self.batch_size, do_shuffle=False, with_label=True) # graph.is_epoch_end值会修改，跳出循环
            emb = self.sess.run(self.H, feed_dict={self.x: mini_batch.X,
                                                   self.z: mini_batch.Z,
                                                   self.x_z: mini_batch.X_Z})   # 得到最后的嵌入
            # emb = self.sess.run(self.net_att_H,
            #                     feed_dict={self.x_z: mini_batch.X_Z
            #                                })

            if train_emb is None:
                train_emb = emb
                train_label = mini_batch.Y
            else:
                train_emb = np.vstack((train_emb, emb)) # 垂直把数组给堆叠起来
                train_label = np.vstack((train_label, mini_batch.Y))

            if graph.is_epoch_end:
                break


        # np.savetxt('./embedding/emb.txt', train_emb, delimiter=' ', fmt='%f')
        # np.savetxt('./embedding/emb_label.txt', train_label, delimiter=' ', fmt='%f')

        np.set_printoptions(suppress=True)  # 不知道是什么


        test_ratio = np.arange(0.5, 1.0, 0.1)
        DANE_dkj = []
        for tr in test_ratio[-1::-1]:
            print('============train ration-{}=========='.format(1 - tr))
            micro, macro = multi_label_classification(train_emb, train_label, tr)
            DANE_dkj.append('{:.4f}'.format(micro) + ' & ' + '{:.4f}'.format(macro))
        print(' & '.join(DANE_dkj))



        acc, nmi = node_clustering(train_emb, train_label)
        print('ACC{:.4f}, NMI {:.4f}'.format(acc, nmi))
        # return acc, nmi
        return micro, macro, DANE_dkj, acc, nmi
        # return train_emb





    def generate_samples(self, graph):
        X = []
        Z = []

        order = np.arange(graph.num_nodes)
        np.random.shuffle(order)  # 打乱顺序

        index = 0
        while True:
            if index > graph.num_nodes:
                break
            if index + self.batch_size < graph.num_nodes:
                mini_batch = graph.sample_by_idx(order[index:index + self.batch_size])
            else:
                mini_batch = graph.sample_by_idx(order[index:])
            index += self.batch_size

            net_H, att_H, net_att_H = self.sess.run([self.net_H, self.att_H, self.net_att_H],
                                         feed_dict={self.x: mini_batch.X,
                                                    self.z: mini_batch.Z,
                                                    self.x_z: mini_batch.X_Z})

            # net_H, att_H = self.sess.run([self.net_H, self.att_H],
            #                              feed_dict={self.x: mini_batch.X,
            #                                         self.z: mini_batch.Z})

            X.extend(0.4*net_H)
            Z.extend(0.6*att_H)

        X = np.array(X) # 转换成数组
        Z = np.array(Z)

        # preprocessing.normalize()函数对指定数据进行转换：
        X = preprocessing.normalize(X, norm='l2') # 正则化（2范式）
        Z = preprocessing.normalize(Z, norm='l2')

        sim = np.dot(X, Z.T) # 矩阵乘法运算
        neg_idx = np.argmin(sim, axis=1)

        # return order
        return order, neg_idx



    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
