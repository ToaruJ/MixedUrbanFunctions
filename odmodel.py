import node2vec as n2v
import networkx as nx
import numpy as np
import gensim.models.callbacks as gscb
import time
import gensim.models.word2vec as w2v


# OD数据(npy文件)的目录，模型保存在这个目录的node2vec/文件夹下
data_root = 'E:/gis_data/出租车OD/'
od_file = data_root + 'od.npy'


class EpochLoggerWV(gscb.CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0
        self.start_time = None

    def on_epoch_begin(self, model):
        self.wv = model.wv.vectors
        self.start_time = time.time()

    def on_epoch_end(self, model):
        print("Epoch %d end. time use: %.2fs, " % (self.epoch, time.time() - self.start_time), end='')
        change = np.max(np.linalg.norm(model.wv.vectors - self.wv, axis=1))
        loss = model.get_latest_training_loss()
        print('change = %.6f, loss = %.6f' % (change, loss))
        self.epoch += 1


def generate_graph_justTAZ(filename):
    '''
    从numpy中读取OD表，生成graph对象(用于仅TAZ无时间的场所表示)
    '''
    od_npy = np.load(filename)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(od_npy.shape[2]))
    edges = []
    data = np.sum(od_npy, axis=(0, 1))
    # 设置阈值为边权重的中位数，仅有OD量较大的边才加入图中
    # threshold1 = np.median(data, axis=0)
    # threshold0 = np.median(data, axis=1)
    threshold1 = np.quantile(data, 0.9, axis=0)
    threshold0 = np.quantile(data, 0.9, axis=1)
    for o in range(data.shape[0]):
        for d in range(data.shape[1]):
            # if data[o, d] > 0:
            if data[o, d] > threshold0[o] or data[o, d] > threshold1[d]:
            # if data[o, d] > threshold0[o]:
                edges.append((o, d, data[o, d]))
    graph.add_weighted_edges_from(edges)
    return graph


def generate_graph_2x12(filename, is_weekday, hour_bin):
    '''
    从numpy中读取OD表，生成graph对象(用于2(周中or周末)x12的场所表示)
    '''
    od_npy = np.load(filename)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(od_npy.shape[2]))
    edges = []
    data = np.sum(od_npy[0:5, :, :, :], axis=0) if is_weekday \
        else np.sum(od_npy[5:7, :, :, :], axis=0)
    for o in range(data.shape[1]):
        for d in range(data.shape[2]):
            if data[hour_bin, o, d] > 0:
                edges.append((o, d, data[hour_bin, o, d]))
    graph.add_weighted_edges_from(edges)
    return graph


def generate_graph_7x12(filename, weekday, hour_bin):
    '''
    从numpy中读取OD表，生成graph对象(用于7x12的场所表示)
    '''
    od_npy = np.load(filename)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(od_npy.shape[2]))
    edges = []
    for o in range(od_npy.shape[2]):
        for d in range(od_npy.shape[3]):
            if od_npy[weekday, hour_bin, o, d] > 0:
                edges.append((o, d, od_npy[weekday, hour_bin, o, d]))
    graph.add_weighted_edges_from(edges)
    return graph


def n2v_train(graph, vec_d, walklen=80, numwalk=80, p=1.0, q=1.0):
    '''
    训练1个node2vec模型
    设定默认参数：window=5
    '''
    n2vwalk = n2v.Node2Vec(graph, dimensions=vec_d, walk_length=walklen, num_walks=numwalk,
                           p=p, q=q, workers=4, seed=1)
    logger = EpochLoggerWV()
    model = n2vwalk.fit(window=5, min_count=1, epochs=50, sample=0,
                        negative=5, callbacks=[logger,], compute_loss=True)
    return model


def n2v_justTAZtrain(filename, model_root, vec_d, walklen=80, numwalk=10, p=1.0, q=1.0):
    '''
    训练node2vec模型，仅有TAZ
    '''
    od_npy = np.load(filename)
    print('OD num:', np.sum(od_npy))
    del od_npy
    graph = generate_graph_justTAZ(filename)
    model = n2v_train(graph, vec_d, walklen, numwalk, p, q)
    vectors = np.zeros(model.wv.vectors.shape)
    for i in range(model.wv.vectors.shape[0]):
        vectors[i, :] = model.wv.get_vector(i)
    np.save(model_root + ('justTAZ_d%d_len%d_num%d_p%.1f_q%.1f.npy' %
            (vec_d, walklen, numwalk, p, q)), vectors)


def n2v_2x12train(filename, model_root, vec_d, walklen=80, numwalk=10, p=1.0, q=1.0):
    '''
    循环训练2*12个node2vec模型
    '''
    od_npy = np.load(filename)
    vectors = np.zeros((2, 12, od_npy.shape[2], vec_d))
    del od_npy
    for i, wday in enumerate([True, False]):
        for h in range(12):
            graph = generate_graph_2x12(filename, wday, h)
            model = n2v_train(graph, vec_d, walklen, numwalk, p, q)
            for taz in range(model.wv.vectors.shape[0]):
                vectors[i, h, taz, :] = model.wv.get_vector(taz)
            print('process: %d/24 finish' % (i * 12 + h + 1))
    np.save(model_root + ('2x12_d%d_len%d_num%d_p%.1f_q%.1f.npy' %
            (vec_d, walklen, numwalk, p, q)), vectors)


def n2v_7x12train(filename, model_root, vec_d, walklen=80, numwalk=10, p=1.0, q=1.0):
    '''
    循环训练7*12个node2vec模型
    '''
    od_npy = np.load(filename)
    vectors = np.zeros((7, 12, od_npy.shape[2], vec_d), dtype=float)
    del od_npy
    for wday in range(7):
        for h in range(12):
            graph = generate_graph_7x12(filename, wday, h)
            model = n2v_train(graph, vec_d, walklen, numwalk, p, q)
            for taz in range(model.wv.vectors.shape[0]):
                vectors[wday, h, taz, :] = model.wv.get_vector(taz)
            print('process: %d/84' % (wday * 12 + h + 1))
    np.save(model_root + ('7x12_d%d_len%d_num%d_p%.1f_q%.1f.npy' %
            (vec_d, walklen, numwalk, p, q)), vectors)


def od_look(filename):
    '''
    TAZ ID = 538是北大
    仅看北大OD，不论北大作为O还是D，其目标流最多都是ID = 503,509,575,526的TAZ
    '''
    taz_id = 553
    od_npy = np.load(filename)
    od_npy = np.sum(od_npy, axis=(0, 1))
    o_index = np.argsort(-od_npy[taz_id, :])
    o_rank = [(o_index[i], od_npy[taz_id, o_index[i]]) for i in range(1, 11)]
    d_index = np.argsort(-od_npy[:, taz_id])
    d_rank = [(d_index[i], od_npy[d_index[i], taz_id]) for i in range(1, 11)]
    print('o:', o_rank)
    print('d:', d_rank)
    destination = 574
    print('from %d to %d: rank = %d, num = %d' % (taz_id, destination, np.argwhere(o_index == destination),
                                                  od_npy[taz_id, destination]))
    print(np.sum(od_npy))


def similar_test(filename):
    '''
    向量的相似性检验
    TAZ ID = 538是北大
    '''
    taz = 574
    vectors = np.load(filename)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # justTAZ的向量
    if vectors.ndim == 2:
        sim = np.matmul(vectors, vectors[taz, :].T)
        index = np.argsort(-sim)
        sim_result = [(index[i], sim[index[i]]) for i in range(1, 11)]
        print(sim_result)

        import pandas as pd
        import os.path as path
        # df = pd.DataFrame({'Id': np.arange(vectors.shape[0]), 'n2v_simq2': sim})
        # df.to_csv(r'E:\各种文档 本四\本科毕设\analyze\n2v_simq2.csv')


if __name__ == '__main__':
    # start_t = time.time()
    n2v_justTAZtrain(od_file, data_root + 'node2vec/版本4v2/',
                     vec_d=16, walklen=80, numwalk=80, p=1, q=0.5)
    n2v_justTAZtrain(od_file, data_root + 'node2vec/版本4v2/',
                     vec_d=16, walklen=80, numwalk=80, p=1, q=2)
    # n2v_2x12train(od_file, data_root + 'node2vec/',
    #               vec_d=80, walklen=80, numwalk=80, p=1, q=1)
    # print(time.time() - start_t)
    # similar_test(data_root + 'node2vec/justTAZ_d64_len80_num80_p1.0_q0.5.npy')
    # similar_test(data_root + 'node2vec/justTAZ_d64_len80_num80_p1.0_q2.0.npy')
    # od_look(od_file)
