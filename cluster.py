import scipy.spatial.distance as SCIdist
import sklearn.mixture as skm
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import skfuzzy.cluster as skf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


taz_f = 'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp'
timeseries_root = 'E:/gis_data/出租车OD/ts_vec/版本1/'
od_root = 'E:/gis_data/出租车OD/node2vec/'
weibo_root = 'E:/gis_data/微博/Sina2016Weibo（处理好的）/doc2vec_vectors/'
bishe_root = 'E:/各种文档 本四/本科毕设/'


def load_file(time_vecf, graph_vecf, weibo_vecf):
    '''
    加载向量文件
    '''
    if type(time_vecf) == str:
        time_vec = np.load(time_vecf)
    else:
        time_vec = [np.load(f) for f in time_vecf]
    if type(graph_vecf) == str:
        graph_vec = np.load(graph_vecf)
    else:
        graph_vec = [np.load(f) for f in graph_vecf]
    weibo_vec = np.load(weibo_vecf)
    return time_vec, graph_vec, weibo_vec


def data_preprocess(time_vec, graph_vec, weibo_vec, del_index=None,
                    regularize=True, pca=None, tsne=None, spectral=None, mask=None):
    '''
    完成向量的初步处理：可能的归一化，PCA主成分变换，最后将不同来源的向量合并
    :param del_index: 需要去除的TAZ（因为数据量过小）
    :param regularize: 是否要将每个文件的向量归一化
    :param pca: None表示无需主成分分析，数字则为PCA的维数（若为-1则表示PCA后维度不变）
    :param tsne: None表示无需T-SNE降维，数字则为T-SNE的维数
    :param spectral: 使用谱聚类、cos相似性的降维方式，数字为降维的维数
    :param mask: 表示是否忽略其中的一些特征向量（使用全1向量代替）
    '''
    # 向量按文件归一化
    def _try_append_regularize(lst, vectors, regularize_, mask_):
        def _zscore(vec):
            v = vec / np.linalg.norm(vec, axis=1, keepdims=True)
            v = v - np.mean(v, axis=0, keepdims=True)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        def _norm1(vec):
            return vec / np.linalg.norm(vec, axis=1, keepdims=True)

        if type(vectors) in (list, tuple):
            if mask_:
                vectors = list(map(lambda v: np.ones_like(v), vectors))
            if regularize_:
                length = len(vectors)
                vectors = map(_norm1, vectors)
                vectors = map(lambda v: v / length, vectors)
            lst.extend(vectors)
        else:
            if mask_:
                vectors = np.ones_like(vectors)
            if regularize_:
                vectors = _norm1(vectors)
            lst.append(vectors)

    if mask is None:
        mask = [False] * 3
    data = []
    for vec, _mask in zip([time_vec, graph_vec, weibo_vec], mask):
        _try_append_regularize(data, vec, regularize, _mask)
    data = np.concatenate(data, axis=1)
    # 去除指定的TAZ
    if del_index is not None:
        data = np.delete(data, del_index, axis=0)
    # 主成分变换、TSNE降维、谱聚类方式降维
    assert (len(list(filter(lambda arg: arg is not None, (pca, tsne, spectral)))) < 2), \
        'cannot use 2 or more of PCA, T-SNE and spectral decomposition'
    if pca is not None:
        if pca < 0:
            pca += data.shape[1] + 1
        pcamodel = PCA(n_components=pca)
        pcamodel.fit(data)
        data = pcamodel.transform(data)
    if tsne is not None:
        tsnemodel = TSNE(n_components=tsne, method='exact', metric='cosine')
        data = tsnemodel.fit_transform(data)
    if spectral is not None:
        if spectral < 0:
            spectral += data.shape[1] + 1
        # 直接拿cos距离矩阵构建权重矩阵。别的方式：cos距离经过RBF一下
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        sim = data @ data.T
        sim[np.diag_indices_from(sim)] = 0
        sqrt_degree = np.sqrt(np.sum(sim, axis=1, keepdims=True))
        sim = -sim / (sqrt_degree * sqrt_degree.T)
        sim[np.diag_indices_from(sim)] = 1
        eig, vector = np.linalg.eig(sim)
        sort_idx = np.argsort(eig)
        data = vector[:, sort_idx[:spectral]]
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
    return data

"""
def onedata_GMM(time_vec, graph_vec, weibo_vec, K, del_index=None, regularize=True,
                pca=None, tsne=None, spectral=None, init=None, mask=None):
    '''
    使用GMM模型的一次数据聚类，输入的是npy向量矩阵
    AIC（A信息准则，越小越好） := 2 * 参数个数K - 2 * ln(似然函数L)
    BIC（贝叶斯信息准则，越小越好）:= 参数个数K * ln(样本数N) - 2 * ln(似然函数L)
    :param regularize: 是否要将每个文件的向量归一化
    :param pca: None表示无需主成分分析，数字则为PCA的维数（若为-1则表示PCA后维度不变）
    :returns: (每样本到每类的概率, AIC, BIC)
    '''
    # 向量归一化
    data = data_preprocess(time_vec, graph_vec, weibo_vec, del_index=del_index,
                           regularize=regularize, pca=pca, tsne=tsne, spectral=spectral, mask=mask)
    # GMM聚类
    gmm = skm.GaussianMixture(n_components=K, covariance_type='diag', verbose=0, verbose_interval=1)
    gmm.fit(data)
    prob = gmm.predict_proba(data)
    aic = gmm.aic(data)
    bic = gmm.bic(data)
    return data, prob, aic, bic
"""

def onedata_FCM(time_vec, graph_vec, weibo_vec, K, del_index=None, regularize=True,
                pca=None, tsne=None, spectral=None, init=None, mask=None):
    '''
    使用FCM模型的一次数据聚类，输入的是npy向量矩阵
    评价指标是FPC（模糊划分系数，1最好）
    :param del_index: 需要去除的TAZ（因为数据量过小）
    :param regularize: 是否要将每个文件的向量归一化
    :param pca: None表示无需主成分分析，数字则为PCA的维数（若为-1则表示PCA后维度不变）
    :returns: (每个对象到每类的模糊C系数, 目标函数迭代过程的变化, FPC)
    '''
    # 向量归一化
    # metric = 'euclidean'
    metric = 'cosine'
    m = 2
    data = data_preprocess(time_vec, graph_vec, weibo_vec, del_index=del_index,
                           regularize=regularize, pca=pca, tsne=tsne, spectral=spectral, mask=mask)
    # FCM模糊聚类，参数m表示模糊程度（越大越模糊）
    if isinstance(init, np.ndarray):
        init = init.T
    if init is not None:
        cntr, u, u0, d, jm, p, fpc = skf.cmeans(data.T, K, m=m, error=0.01,
                                            maxiter=10, metric=metric, init=init)
    else:
        # 可用seed：不去除数据过少的TAZ(time series 7x24)：(vsize=128*2+80)332757414、983499139。
        #   (time series 2x24)：(vsize=64*3)371938383, (vsize=96*2+80)
        #   7*24: 786198485
        seed = np.random.randint(0, 1024 * 1024 * 1024)
        # print('seed:', seed)
        # seed = 371938383

        # 用硬聚类K-Means作为初始聚类中心
        kmeans = KMeans(K)
        kmeans.fit(data)
        d = SCIdist.cdist(data, kmeans.cluster_centers_, metric=metric).T
        d = np.fmax(d, np.finfo(np.float64).eps)
        from skfuzzy.cluster.normalize_columns import normalize_power_columns
        init = normalize_power_columns(d, - 2. / (m - 1))

        cntr, u, u0, d, jm, p, fpc = skf.cmeans(data.T, K, m=m, error=0.01,
                                                maxiter=10, metric=metric, init=init)
    # u_, u0_, d_, jm_, p_, fpc_ = skf.cmeans_predict(data.T, cntr, m=m, metric=metric, error=1e-3, maxiter=15)
    return data, u.T, jm, fpc

"""
def gaussian_cluster(time_vecf, graph_vecf, weibo_vecf, K, timetype: str,
                     del_index=None, regularize=True, pca=None, tsne=None,
                     spectral=None, init=None, mask=mask):
    '''
    对向量进行高斯混合模型聚类，输出高斯聚类下各个TAZ属于各类型的概率
    输入的是向量文件路径
    :param graph_vecf: 可以是一个文件(str)或文件列表list()
    :param K: 聚类数
    :param timetype: str: 'justTAZ', '2x12'其中一种
    '''
    time_vec, graph_vec, weibo_vec = load_file(time_vecf, graph_vecf, weibo_vecf)
    X, probs, aics, bics = None, None, None, None

    if timetype.lower() == 'justtaz':
        X, probs, aics, bics = onedata_GMM(time_vec, graph_vec, weibo_vec, K, del_index=del_index,
                                      regularize=regularize, pca=pca, tsne=tsne, spectral=spectral, init=init)
        print({'aic': aics, 'bic': bics})

    elif timetype.lower() == '2x12':
        X = []
        probs = np.zeros((2, 12, time_vec.shape[0], K), dtype=float)
        aics = np.zeros((2, 12), dtype=float)
        bics = np.zeros((2, 12), dtype=float)
        for i in range(2):
            for h in range(12):
                odvec = list(map(lambda v: v[i, h, :, :], graph_vec))
                x_, prob, aic, bic = onedata_GMM(time_vec, odvec, weibo_vec[i, h, :, :],
                                             K, del_index=del_index, regularize=regularize,
                                             pca=pca, tsne=tsne, spectral=spectral, init=init)
                probs[i, h, :, :] = prob
                aics[i, h] = aic
                bics[i, h] = bic
                X.append(x_)
        X = np.stack(X, axis=0)
    return X, probs, aics, bics
"""

def FCM_cluster(time_vecf, graph_vecf, weibo_vecf, K, timetype: str, del_index=None,
                regularize=True, pca=None, tsne=None, spectral=None, init=None, mask=None):
    '''
    对向量进行模糊C平均聚类，输出模糊聚类下各个TAZ属于各类型的概率
    输入的是向量文件路径
    :param graph_vecf: 可以是一个文件(str)或文件列表list()
    :param K: 聚类数
    :param timetype: str: 'justTAZ', '2x12'其中一种
    :param del_index: 需要去除的TAZ（因为数据量过小）
    '''
    time_vec, graph_vec, weibo_vec = load_file(time_vecf, graph_vecf, weibo_vecf)
    X, probs, jms, fpcs = None, None, None, None

    if timetype.lower() == 'justtaz':
        X, probs, jms, fpcs = onedata_FCM(time_vec, graph_vec, weibo_vec, K, del_index=del_index,
                                          regularize=regularize, pca=pca, tsne=tsne,
                                          spectral=spectral, init=init, mask=mask)

    elif timetype.lower() == '2x12':
        X = []
        probs = np.zeros((2, 12, time_vec.shape[0], K))
        fpcs = np.zeros((2, 12))
        jms = []
        for i in range(2):
            for h in range(12):
                odvec = list(map(lambda v: v[i, h, :, :], graph_vec))
                x_, prob, jm, fpc = onedata_FCM(time_vec, odvec, weibo_vec[i, h, :, :],
                                                K, del_index=del_index, regularize=regularize,
                                                pca=pca, tsne=tsne, spectral=spectral, mask=mask)
                probs[i, h, :, :] = prob
                jms.append(jm)
                fpcs[i, h] = fpc
                X.append(x_)
        X = np.stack(X, axis=0)
    return X, probs, jms, fpcs


def watch_distrib(time_vecf, graph_vecf, weibo_vecf, regularize=True):
    '''
    散点图观察向量的2维分布，使用PCA降维或TSNE
    '''
    time_vec, graph_vec, weibo_vec = load_file(time_vecf, graph_vecf, weibo_vecf)
    # PCA方式降维
    # data = data_preprocess(o_vec, d_vec, od_vec, weibo_vec, regularize, pca=2)
    # tsne方式降维
    data = data_preprocess(time_vec, graph_vec, weibo_vec, regularize, tsne=2)
    print(data.shape)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

"""
def draw_aic_plot():
    '''
    测试不同的分类数，计算AIC并绘制曲线
    '''
    aics = [gaussian_cluster([timeseries_root + 'o7x24_d80_ep10.npy',
                             timeseries_root + 'd7x24_d80_ep10.npy'],
                             [od_root + 'justTAZ_d80_len80_num80_p1.0_q0.5.npy',
                              od_root + 'justTAZ_d80_len80_num80_p1.0_q2.0.npy'],
                             weibo_root + 'justTAZ_size80_ep20.npy',
                             K=k, timetype='justTAZ', regularize=True)[1]
            for k in range(2, 9)]
    aics = np.array(aics) / 1000
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 9), aics)
    plt.title('AIC metrics of different number of clusters in GMM', size=14)
    plt.xlabel('num of clusters', size=14)
    plt.ylabel('AIC (*$10^3$)', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()
"""

def FCM_sample(K, poi_csv, opt_k=None, suffix='', datanum_stat=None, mask=None):
    '''
    使用FCM聚类的一个实例。
    可能筛选：去除OD数 < 600、微博数 < 600的TAZ
    :param datanum_stat: OD数、微博数统计表。可能需要去除OD数、微博数过少的TAZ
    :param opt_k: 最优的聚类数
    '''
    # from gdPOI import poi_mytype_en as typeen
    from globalval import landuse_name_v2 as typeen
    poidf = pd.read_csv(poi_csv)
    del_index = None

    # 对比先去除稀疏TAZ数据再聚类，以及先聚类再去除稀疏数据回归，如果是先聚类效果好，则需注释下方代码
    if datanum_stat is not None:
        numstat = pd.read_csv(datanum_stat)
        poidf = pd.merge(poidf, numstat, on='Id')
        dropid = poidf.loc[(poidf['o_num'] < 600) | (poidf['d_num'] < 600)
                            | (poidf['weibo_num'] < 600), 'Id']
        del_index = np.array(dropid.index)
        poidf.drop(index=dropid.index, inplace=True)

    print(poidf.shape[0])

    # 初始化很关键，有2类初始化：随机种子；以TAZ中各类POI的占比作为初始隶属度
    # init_list = [poidf[typeen[0]],#+ poidf[typeen[2]],
    #             # np.zeros(poidf.shape[0]) + 1,
    #              poidf[typeen[1]] + poidf[typeen[2]],
    #              poidf[typeen[3]] + poidf[typeen[4]],
    #              poidf[typeen[6]],
    #              poidf[typeen[5]] + poidf[typeen[7]]]
    init_list = [poidf[typeen[0]],  # + poidf[typeen[2]],
                 # np.zeros(poidf.shape[0]) + 1,
                 poidf[typeen[1]],
                 poidf[typeen[2]] + poidf[typeen[3]],
                 poidf[typeen[6]],
                 poidf[typeen[4]] + poidf[typeen[5]]]
    init = np.stack(init_list, axis=1)
    # init = init / np.sum(init, axis=1, keepdims=True)

    while True:
        X, probs, jms, fpcs = FCM_cluster(timeseries_root + 'od7x24abs_cnn_triplet_d32_ep100_ml4.npy',
                                          [od_root + '版本4/justTAZ_d16_len80_num80_p1.0_q0.5.npy',
                                           od_root + '版本4/justTAZ_d16_len80_num80_p1.0_q2.0.npy'],
                                          # od_root + 'justTAZ_d32_len80_num80_p1.0_q2.0.npy',
                                          weibo_root + 'justTAZ_size32_ep40.npy', del_index=del_index,
                                          K=K, timetype='justTAZ', regularize=True,
                                          pca=None, tsne=None, spectral=None, init=None, mask=mask)

        # print(jms)
        # print(probs)
        # 绘制训练过程的loss曲线
        # plt.figure(figsize=(8, 6))
        # plt.plot(np.arange(jms.shape[0]) + 1, jms / 1000)
        # plt.title('objective function loss in FCM epochs', size=14)
        # plt.xlabel('iter of epochs', size=14)
        # plt.ylabel('objective function loss ($\\times10^3$)', size=14)
        # plt.xticks(size=14)
        # plt.yticks(size=14)
        # plt.show()

        # 混合熵

        if K != opt_k:
            break
        entropy = np.sum(-probs * np.log2(probs), axis=1)
        # 和TAZ的连接表
        data = np.concatenate([poidf['Id'].values.reshape(poidf.shape[0], 1), probs,
                               entropy.reshape((probs.shape[0], 1))], axis=1)
        frame = pd.DataFrame(data, columns=['Id'] + ['cluster%d' % (i + 1) for i in range(probs.shape[1])] +
                             ['entropy'])
        frame['main_cluster'] = np.argmax(probs, axis=1) + 1
        frame['Id'] = frame['Id'].astype(int)

        # 聚完类后去除稀疏数据
        if datanum_stat is not None:
            numstat = pd.read_csv(datanum_stat)
            frame = pd.merge(frame, numstat, on='Id')
            dropid = frame.loc[(frame['o_num'] < 600) | (frame['d_num'] < 600)
                                | (frame['weibo_num'] < 600), 'Id']
            frame.drop(index=dropid.index, inplace=True)

        filename = bishe_root + 'cluster/fcm_k%d' % K
        filename = filename if suffix == '' else filename + '_%s' % suffix
        # frame.to_csv(filename + '.csv', index=False)

        cluster = frame
        pois = pd.read_csv(poi_csv)
        merge = pd.merge(cluster, pois, on='Id')
        # 相关系数阵
        # poiscore = merge[gdPOI.poi_mytype_en]
        poiscore = merge[typeen]
        clusterscore = merge[['cluster%d' % (i + 1) for i in range(K)]]
        corr = np.corrcoef(np.concatenate([clusterscore, poiscore], axis=1), rowvar=False)
        corr = corr[0:K, K:]
        argmax = np.argmax(corr, axis=1)
        if len(set(argmax)) == K:
            break
        break

    # 计算轮廓系数，FPC系数（针对模糊分类），确定效果最好的聚类数
    sil_score = silhouette_score(X, np.argmax(probs, axis=1), metric='cosine')
    print(f'K = {K}: silhouette {sil_score}, fpc {fpcs}')
    # 后续分析
    if K == opt_k:
        # 综合向量的散点图
        tsnemodel = TSNE(metric='cosine')
        scatter2d = tsnemodel.fit_transform(X)
        plt.figure(dpi=125)
        plt.scatter(scatter2d[:, 0], scatter2d[:, 1])
        plt.title('scatter of vectors (t-SNE)')
        plt.show()
        print('jsm', jms)

        from postanalyze import LinearAnalyzePOI, drawCluster, LinearAnalyzeDist
        LinearAnalyzePOI(frame, poi_csv, K, min_sum=00)
        LinearAnalyzeDist(frame, bishe_root + 'analyze/poitaz_tfidfv2.csv',
                          bishe_root + 'analyze/odata_dist_stat.csv',
                          bishe_root + 'analyze/ddata_dist_stat.csv', min_sum = 00)
        colormap = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
        for i in range(K):
            drawCluster(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
                        frame, column='cluster%d' % (i + 1), color=colormap[i],
                        title='cluster%d' % (i + 1))
    return fpcs, sil_score


"""
def gaussian_sample(K, poi_csv, suffix='', datanum_stat=None):
    '''
    使用FCM聚类的一个实例。
    可能筛选：去除OD数 < 600、微博数 < 600的TAZ
    :param datanum_stat: OD数、微博数统计表。可能需要去除OD数、微博数过少的TAZ
    '''
    opt_k = 5   # 最优的聚类数
    # from gdPOI import poi_mytype_en as typeen
    from gdPOI import landuse_name_v2 as typeen
    poidf = pd.read_csv(poi_csv)
    del_index = None

    # 对比先去除稀疏TAZ数据再聚类，以及先聚类再去除稀疏数据回归，结果是先聚类效果好，因此下方代码注释了
    if datanum_stat is not None:
        numstat = pd.read_csv(datanum_stat)
        poidf = pd.merge(poidf, numstat, on='Id')
        dropid = poidf.loc[(poidf['o_num'] < 600) | (poidf['d_num'] < 600)
                            | (poidf['weibo_num'] < 600), 'Id']
        del_index = np.array(dropid.index)
        poidf.drop(index=dropid.index, inplace=True)

    print(poidf.shape[0])

    # 初始化很关键，有2类初始化：随机种子；以TAZ中各类POI的占比作为初始隶属度
    # init_list = [poidf[typeen[0]],#+ poidf[typeen[2]],
    #             # np.zeros(poidf.shape[0]) + 1,
    #              poidf[typeen[1]] + poidf[typeen[2]],
    #              poidf[typeen[3]] + poidf[typeen[4]],
    #              poidf[typeen[6]],
    #              poidf[typeen[5]] + poidf[typeen[7]]]
    init_list = [poidf[typeen[0]],  # + poidf[typeen[2]],
                 # np.zeros(poidf.shape[0]) + 1,
                 poidf[typeen[1]],
                 poidf[typeen[2]] + poidf[typeen[3]],
                 poidf[typeen[6]],
                 poidf[typeen[4]] + poidf[typeen[5]]]
    init = np.stack(init_list, axis=1)
    # init = init / np.sum(init, axis=1, keepdims=True)

    while True:
        X, probs, aic, bic = gaussian_cluster(timeseries_root + 'od7x24abs_cnn_triplet_d32_ep100_ml4.npy',
                                       [od_root + 'justTAZ_d16_len80_num80_p1.0_q0.5.npy',
                                        od_root + 'justTAZ_d16_len80_num80_p1.0_q2.0.npy'],
                                       # od_root + 'justTAZ_d32_len80_num80_p1.0_q0.5.npy',
                                       weibo_root + 'justTAZ_size32_ep40.npy', del_index=del_index,
                                       K=K, timetype='justTAZ', regularize=True, pca=1, tsne=None)
        print(probs)

        # 混合熵

        if K != opt_k:
            break
        entropy = np.sum(-probs * np.log2(probs), axis=1)
        # 和TAZ的连接表
        data = np.concatenate([poidf['Id'].values.reshape(poidf.shape[0], 1), probs,
                               entropy.reshape((probs.shape[0], 1))], axis=1)
        frame = pd.DataFrame(data, columns=['Id'] + ['cluster%d' % (i + 1) for i in range(probs.shape[1])] +
                             ['entropy'])
        frame['main_cluster'] = np.argmax(probs, axis=1) + 1
        frame['Id'] = frame['Id'].astype(int)

        # 聚完类后去除稀疏数据
        if datanum_stat is not None:
            numstat = pd.read_csv(datanum_stat)
            frame = pd.merge(frame, numstat, on='Id')
            dropid = frame.loc[(frame['o_num'] < 600) | (frame['d_num'] < 600)
                                | (frame['weibo_num'] < 600), 'Id']
            frame.drop(index=dropid.index, inplace=True)

        filename = bishe_root + 'cluster/fcm_k%d' % K
        filename = filename if suffix == '' else filename + '_%s' % suffix
        # frame.to_csv(filename + '.csv', index=False)

        cluster = frame
        pois = pd.read_csv(poi_csv)
        merge = pd.merge(cluster, pois, on='Id')
        # 相关系数阵
        # poiscore = merge[gdPOI.poi_mytype_en]
        poiscore = merge[typeen]
        clusterscore = merge[['cluster%d' % (i + 1) for i in range(K)]]
        corr = np.corrcoef(np.concatenate([clusterscore, poiscore], axis=1), rowvar=False)
        corr = corr[0:K, K:]
        argmax = np.argmax(corr, axis=1)
        if len(set(argmax)) == K:
            break
        break

    # 计算轮廓系数，AIC和BIC系数（针对高斯分类），确定效果最好的聚类数
    sil_score = silhouette_score(X, np.argmax(probs, axis=1), metric='cosine')
    print(f'K = {K}: silhouette {sil_score}, aic {aic}, bic {bic}')
    # 后续分析
    if K == opt_k:
        # 综合向量的散点图
        tsnemodel = TSNE(metric='cosine')
        scatter2d = tsnemodel.fit_transform(X)
        plt.figure(dpi=125)
        plt.scatter(scatter2d[:, 0], scatter2d[:, 1])
        plt.title('scatter of vectors (t-SNE)')
        plt.show()

        from postanalyze import LinearAnalyzePOI, drawCluster, LinearAnalyzeDist
        LinearAnalyzePOI(frame, poi_csv, K, min_sum=00)
        LinearAnalyzeDist(frame, bishe_root + 'analyze/poitaz_tfidfv2.csv',
                          bishe_root + 'analyze/odata_dist_stat.csv',
                          bishe_root + 'analyze/ddata_dist_stat.csv', min_sum = 00)
        colormap = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
        for i in range(K):
            drawCluster(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
                        frame, column='cluster%d' % (i + 1), color=colormap[i],
                        title='cluster%d' % (i + 1))
    return aic, bic
"""

def FCM_cluster_try(filelist, poi_csv, K, m, regularize=True, pca=None, tsne=None,
                    metric='euclidean', min_sum=0):
    '''
    对向量进行模糊C平均聚类，输出模糊聚类下各个TAZ属于各类型的概率
    输入的是向量文件路径，该测试函数可以指定使用哪些向量
    :param filelist: 文件列表list(str)
    :param K: 聚类数
    '''
    datalist = [np.load(f) for f in filelist]
    # 向量归一化
    if regularize:
        datalist = [item / np.linalg.norm(item, axis=1).reshape(item.shape[0], 1)
                    for item in datalist]
    data = np.concatenate(datalist, axis=1)
    # 主成分变换或TSNE
    assert (pca is None or tsne is None), 'cannot use both PCA and T-SNE'
    if pca is not None:
        if pca < 0:
            pca += data.shape[1] + 1
        pcamodel = PCA(n_components=pca)
        pcamodel.fit(data)
        data = pcamodel.transform(data)
    if tsne is not None:
        tsnemodel = TSNE(n_components=tsne)
        data = tsnemodel.fit_transform(data)
    cntr, u, u0, d, jm, p, fpc = skf.cmeans(data.T, K, m=m, error=1e-3,
                                            maxiter=15, metric=metric)
    prob = u.T
    print(fpc)
    print(jm)
    print(prob)
    # 混合熵
    entropy = np.sum(-prob * np.log2(prob), axis=1)
    # 和TAZ的shapefile连接表
    data = np.concatenate([np.arange(prob.shape[0]).reshape((prob.shape[0], 1)), prob,
                           entropy.reshape((prob.shape[0], 1))], axis=1)
    frame = pd.DataFrame(data, columns=['Id'] + ['cluster%d' % (i + 1) for i in range(prob.shape[1])] +
                                       ['entropy'])
    frame['Id'] = frame['Id'].astype(int)


if __name__ == '__main__':
    # GMM的分类效果不好，放弃了
    # probs_min = None
    # aics_min = 0
    # for i in range(10):
    #     probs, aics, bics = gaussian_cluster(timeseries_root + 'o7x24_d80_ep10.npy',
    #                                    timeseries_root + 'd7x24_d80_ep10.npy',
    #                                    [od_root + 'justTAZ_d80_len80_num80_p1.0_q0.5.npy',
    #                                    od_root + 'justTAZ_d80_len80_num80_p1.0_q2.0.npy'],
    #                                    weibo_root + 'justTAZ_size80_ep20.npy',
    #                                    K=K, timetype='justTAZ', regularize=True)
    #     if aics < aics_min:
    #         aics_min = aics
    #         probs_min = probs
    # print(probs_min)
    # frame = pd.DataFrame(probs_min, columns=['cluster%d' % (i + 1) for i in range(K)])
    # frame['Id'] = np.arange(frame.shape[0])
    # frame.to_csv(bishe_root + 'GMM_K5.csv', index=False)

    # probs, aics, bics = gaussian_cluster(timeseries_root + 'o7x24_r2dist_d64_ep10.npy',
    #                                      timeseries_root + 'd7x24_r2dist_d64_ep10.npy',
    #                                      [od_root + 'justTAZ_d64_len80_num80_p1.0_q0.5.npy',
    #                                       od_root + 'justTAZ_d64_len80_num80_p1.0_q2.0.npy'],
    #                                      weibo_root + 'justTAZ_size80_ep20.npy',
    #                                      K=5, timetype='justTAZ', regularize=True, pca=None)
    # print(probs)
    # print(bics)
    '''
    fpc, sil_score = FCM_sample(5, bishe_root + 'analyze/landuse_stat2.csv', suffix='triplet_7x24_final',
                                datanum_stat=bishe_root + 'analyze/datanum_stat.csv', opt_k=5)
    fpcs = []
    sil_scores = []
    x = range(2, 21)
    for loop in range(10):
        fpcx = []
        sil_scorex = []
        for i in x:
            fpc, sil_score = FCM_sample(i, bishe_root + 'analyze/landuse_stat2.csv', suffix='triplet_7x24_final',
                                        datanum_stat=bishe_root + 'analyze/datanum_stat.csv')
            fpcx.append(fpc)
            sil_scorex.append(sil_score)
        fpcs.append(fpcx)
        sil_scores.append(sil_scorex)
    fpcs = np.array(fpcs)
    sil_scores = np.array(sil_scores)
    # frame = pd.DataFrame(fpcs, columns=['K_%d' % k for k in x])
    # frame.to_csv(bishe_root + 'analyze/fpc_values.csv', index=False)
    # frame = pd.DataFrame(sil_scores, columns=['K_%d' % k for k in x])
    # frame.to_csv(bishe_root + 'analyze/silscore_values.csv', index=False)
    mean_fpcs = np.mean(fpcs, axis=0)
    yerror = np.stack([np.max(fpcs, axis=0) - mean_fpcs,
                       mean_fpcs - np.min(fpcs, axis=0)], axis=0)
    plt.figure(dpi=125)
    plt.errorbar(x, mean_fpcs, yerr=yerror)
    mean_silscores = np.mean(sil_scores, axis=0)
    yerror = np.stack([np.max(sil_scores, axis=0) - mean_silscores,
                       mean_silscores - np.min(sil_scores, axis=0)], axis=0)
    plt.errorbar(x, mean_silscores, yerr=yerror)
    # plt.fill_between(x, np.min(fpcs, axis=0), np.max(fpcs, axis=0), alpha=0.2)
    # fpcs_diff = np.diff(mean_fpcs, n=2)
    # plt.plot(range(3, fpcs_diff.shape[0] + 3), fpcs_diff)
    plt.title('fpc score and 2-order derivative of fpc score')
    plt.show()
    '''
    for i in range(3):
        mask = [True] * 3
        mask[i] = False
        fpc, sil_score = FCM_sample(5, bishe_root + 'analyze/landuse_stat2.csv', suffix=f'ablation_{i}',
                                    datanum_stat=bishe_root + 'analyze/datanum_stat.csv', opt_k=5,
                                    mask=mask)

    # FCM_cluster_try([timeseries_root + 'o7x24_d80_ep10.npy',
    #                  timeseries_root + 'd7x24_d80_ep10.npy',
    #                  weibo_root + 'justTAZ_size80_ep20.npy'],
    #                 bishe_root + 'poitaz_stat.csv', K=5, m=1.5,
    #                 regularize=True, pca=None, tsne=None, metric='euclidean', min_sum=100)

    # watch_distrib(timeseries_root + 'o7x24_d80_ep10.npy',
    #               timeseries_root + 'd7x24_d80_ep10.npy',
    #               [od_root + 'justTAZ_d80_len80_num80_p1.0_q0.5.npy',
    #               od_root + 'justTAZ_d80_len80_num80_p1.0_q2.0.npy'],
    #               weibo_root + 'justTAZ_size80_ep20.npy')

    # 验证每个向量单独聚类时的轮廓系数
    # vec = np.load(timeseries_root + 'od7x24abs_cnn_triplet_d32_ep100_ml4.npy')
    # vec = np.load(od_root + 'justTAZ_d16_len80_num80_p1.0_q2.0.npy')
    # vec = np.load(weibo_root + 'justTAZ_size32_ep40.npy')
    # vec = np.concatenate([
    #     np.load(od_root + 'justTAZ_d16_len80_num80_p1.0_q0.5.npy'),
    #     np.load(od_root + 'justTAZ_d16_len80_num80_p1.0_q2.0.npy')
    # ], axis=1)
    # for K in range(2, 10):
    #     kmeans = KMeans(K)
    #     kmeans.fit(vec)
    #     sil_score = silhouette_score(vec, kmeans.labels_, metric='cosine')
    #     print(f'K = {K}: silhouette {sil_score}')
