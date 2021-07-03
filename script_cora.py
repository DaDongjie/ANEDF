from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import linecache
from Dataset.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
from Utils import gpu_info
import os
import random
import tensorflow as tf
from Utils.utils import write_embedding

if __name__=='__main__':

    # gpus_to_use, free_memory = gpu_info.get_free_gpu()
    # print(gpus_to_use, free_memory)
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    random.seed(9001)


    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    sess = tf.Session(config=config)



    dataset_config = {'feature_file': './Database/cora/features.txt',
                      'graph_file': './Database/cora/edges.txt',
                      'walks_file': './Database/cora/walks.txt',
                      'label_file': './Database/cora/group.txt'}
    graph = Dataset(dataset_config)

    pretrain_config = {
        'net_shape': [256, 128],
        'att_shape': [256, 128],
        'net_att_shape': [512, 256],

        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'net_att_input_dim': graph.num_net_atts,

        'pretrain_params_path': './Log/cora/pretrain_params.pkl'
    }

    model_config = {
        'net_shape': [256, 128],
        'att_shape': [256, 128],
        'net_att_shape': [512, 256],

        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'net_att_input_dim': graph.num_net_atts,

        'is_init': True,

        'pretrain_params_path': './Log/cora/pretrain_params.pkl'
    }

    trainer_config = {
        'net_shape': [256, 128],
        'att_shape': [256, 128],
        'net_att_shape': [512, 256],

        'net_input_dim': graph.num_nodes,
        'att_input_dim': graph.num_feas,
        'net_att_input_dim': graph.num_net_atts,

        'drop_prob': 0.2,
        'learning_rate': 1e-5,
        'batch_size': 100,
        'num_epochs': 500,  #
        'beta': 100,# 100
        'alpha': 50,
        'gamma': 500,
        'sita': 300,
        'model_path': './Log/cora/cora_model.pkl',
    }

    pretrainer = PreTrainer(pretrain_config) # 实例化PreTrainer类
    pretrainer.pretrain(graph.X, 'net') # 结构  低维表示
    pretrainer.pretrain(graph.Z, 'att') # 属性  低维表示
    pretrainer.pretrain(graph.X_Z, 'net_att') # 结构+属性  低维表示

    # 利用网络和属性进行预训练，利用预训练创建模型
    model = Model(model_config)
    trainer = Trainer(model, trainer_config)
    trainer.train(graph)
    micro, macro, dane_vae, acc, nmi = trainer.infer(graph)
    # write_embedding(embedding_result, './embedding/cora.embed')

    # DANE_dkj = []
    # flag = False
    # with open("./result/cora/cora.txt", "w") as f:
    #     for beta_i in [1, 10, 50, 100, 200, 500, 1000]:
    #         for alpha_i in [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500]:
    #             for gama_i in [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500]:
    #                 # for i in range(20):
    #                 #     for beta_i in [50]:
    #                 #         for alpha_i in [50]:
    #                 #             for gama_i in [1]:
    #
    #                 tf.reset_default_graph()
    #                 trainer_config = {
    #                     'net_shape': [256, 128],
    #                     'att_shape': [256, 128],
    #                     'net_att_shape': [512, 256],
    #
    #                     'net_input_dim': graph.num_nodes,
    #                     'att_input_dim': graph.num_feas,
    #                     'net_att_input_dim': graph.num_net_atts,
    #
    #                     'drop_prob': 0.2,
    #                     'learning_rate': 1e-5,
    #                     'batch_size': 100,
    #                     'num_epochs': 500,  #
    #                     # 'batch_size': 64,  # 256
    #
    #                     'beta': beta_i - 1,
    #                     'alpha': alpha_i,
    #                     'gamma': gama_i,
    #
    #                     'model_path': './Log/cora/cora_model.pkl',
    #                 }
    #                 if flag:
    #                     pretrainer = PreTrainer(pretrain_config)
    #                     pretrainer.pretrain(graph.X, 'net')
    #                     pretrainer.pretrain(graph.Z, 'att')
    #                     pretrainer.pretrain(graph.X_Z, 'net_att')
    #                     flag = False
    #
    #                 model = Model(model_config)
    #                 trainer = Trainer(model, trainer_config)
    #                 trainer.train(graph)
    #                 micro, macro, dane_vae, acc, nmi = trainer.infer(graph)
    #                 result_single = 'beta_i={:d}'.format(beta_i) + ' & alpha_i={:.4f}'.format(
    #                     alpha_i) + ' & gama_i={:.4f}'.format(gama_i) + ' & micro={:.4f}'.format(
    #                     micro) + ' & ' + 'macro={:.4f}'.format(macro) + ' & ' + ' & '.join(
    #                     dane_vae) + ' & ' + ' & ACC={:.4f}'.format(
    #                     acc) + ' & ' + 'NMI={:.4f}'.format(nmi)
    #                 # acc, nmi = trainer.infer(graph)
    #                 # result_single = 'beta_i={:.4f}'.format(beta_i) + ' & alpha_i={:.4f}'.format(
    #                 #     alpha_i) + ' & gama_i={:.4f}'.format(gama_i) + ' & ACC={:.4f}'.format(
    #                 #     acc) + ' & ' + 'NMI={:.4f}'.format(nmi)
    #
    #                 DANE_dkj.append(result_single)
    #
    #                 f.write(result_single + '\n')
    #                 f.flush()



    # pretrainer = PreTrainer(pretrain_config)  # 实例化PreTrainer类
    # pretrainer.pretrain(graph.X, 'net')  # 结构  低维表示
    # pretrainer.pretrain(graph.Z, 'att')  # 属性  低维表示
    # pretrainer.pretrain(graph.X_Z, 'net_att')  # 结构+属性  低维表示

    # DANE_dkj = []
    # Flag = False
    # with open("./result/cora/cora123.txt", "a") as f:
    #     for i in range(10):
    #         tf.reset_default_graph()
    #         trainer_config = {
    #             'net_shape': [256, 128],
    #             'att_shape': [256, 128],
    #             'net_att_shape': [512, 256],
    #
    #             'net_input_dim': graph.num_nodes,
    #             'att_input_dim': graph.num_feas,
    #             'net_att_input_dim': graph.num_net_atts,
    #             'drop_prob': 0.2,
    #             'learning_rate': 1e-5,
    #             'batch_size': 100,
    #             'num_epochs': 500,  #
    #             'beta': 100,# 100
    #             'alpha': 50,
    #             'gamma': 500,
    #             'sita': 300,
    #             'model_path': './Log/cora/cora_model.pkl',
    #         }
    #
    #         if Flag:
    #             pretrainer = PreTrainer(pretrain_config)
    #             pretrainer.pretrain(graph.X, 'net')
    #             pretrainer.pretrain(graph.Z, 'att')
    #             pretrainer.pretrain(graph.X_Z, 'net_att')
    #             Flag = False
    #
    #         model = Model(model_config)
    #         trainer = Trainer(model, trainer_config)
    #         trainer.train(graph)
    #         micro, macro, dane_dkj, acc, nmi = trainer.infer(graph)
    #         result_single = 'i={:d}'.format(i) + ' & ' + ' & '.join(
    #              dane_dkj) + ' & ACC={:.4f}'.format(
    #                     acc) + ' & ' + 'NMI={:.4f}'.format(nmi)
    #
    #         # result_single = 'alpha={:.4f}'.format(trainer_config['alpha']) +' beta={:.4f}'.format(trainer_config['beta']) \
    #         #     +' gamma={:.4f}'.format(trainer_config['gamma']) + 'i={:d}'.format(i) + ' & micro={:.4f}'.format(
    #         #     micro) + ' & ' + 'macro={:.4f}'.format(macro) + ' & '.join(
    #         #     dane_vae) + ' & ' + ' & ACC={:.4f}'.format(acc) + ' & ' + 'NMI={:.4f}'.format(nmi)
    #
    #         DANE_dkj.append(result_single)
    #
    #         f.write(result_single + '\n')
    #         f.flush()
    #
