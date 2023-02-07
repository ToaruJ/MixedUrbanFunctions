import numpy as np
import scipy.special as spe
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os.path as path
from sklearn.linear_model import LinearRegression
import geopandas as gpd


bishe_root = 'E:/各种文档 本四/本科毕设/'


# 逐步回归分析部分代码
def StepRegress(data: np.ndarray, FstarProp):
    '''
    逐步回归分析的主代码
    :param data: numpy数组形式数据[[x1 x2 x3 ... y], [样本2], ...]，最后一维是y
    :param FstarProp: 逐步回归的F检验值标准概率，仅保留高于该概率值的变量
    '''
    # 相关系数矩阵
    relation_R = np.corrcoef(data, rowvar=False)
    print('各变量相关系数：', np.round(relation_R[-1, :-1], 4))
    avg = np.mean(data, axis=0)       # 各变量平均值
    stdev = np.std(data, axis=0)      # 各变量的标准差

    var_list = np.zeros(data.shape[1] - 1, dtype=np.bool_)
    # 开始逐步分析循环
    newVar_index = TryAddVar(relation_R, data.shape[0], var_list, 0.9)
    while newVar_index is not None:
        try_del = True
        while try_del:
            try_del = TryDelVar(relation_R, data.shape[0], var_list, newVar_index, FstarProp)
        newVar_index = TryAddVar(relation_R, data.shape[0], var_list, FstarProp)

    # 回归后的分析，包括计算复相关系数、回归参数、方程显著性检验
    correlation = np.sqrt(1 - relation_R[-1, -1])
    print('复相关系数：%.4f' % correlation)
    bn = [None for i in range(var_list.shape[0])]   # 记录原始方程的回归系数
    b0 = avg[-1]                                # 记录原始方程的常数项
    # 计算原回归方程
    for i in range(var_list.shape[0]):
        if var_list[i]:
            bn[i] = stdev[-1] / stdev[i] * relation_R[i, -1]
            b0 -= avg[i] * bn[i]
    print('回归方程：y = %.4f' % b0, end='')
    for i in range(len(bn)):
        if bn[i] is not None:
            print(' + ' if bn[i] > 0 else ' - ', '%.4f' % abs(bn[i]),
                  'x%d' % (i + 1), sep='', end='')
    print('\n显著性：', end='')
    var_num = var_list.nonzero()[0].shape[0]
    for i in range(len(bn)):
        if bn[i] is not None:
            partial_reg = relation_R[i, -1] * relation_R[i, -1] / relation_R[i, i]
            Ftest = partial_reg / relation_R[-1, -1] * (data.shape[0] - var_num - 1)
            # F检验小于临界值
            Fprop = spe.fdtr(1, data.shape[0] - var_num - 1, Ftest)
            print('%.4f,' % Fprop, end='')
    print('')


def TransFormR(R: np.ndarray, index):
    '''
    对相关系数矩阵R变换。用于逐步回归分析
    :param R: 相关系数矩阵
    :param index: 要更改的下标
    :return: 变换后的R
    '''
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if i != index and j != index:
                R[i, j] -= R[i, index] * R[index, j] / R[index, index]
    for i in range(R.shape[0]):
        if i != index:
            R[index, i] /= R[index, index]
            R[i, index] /= -R[index, index]
    R[index, index] = 1 / R[index, index]
    return R


def TryDelVar(R: np.ndarray, sample_num, var_list: np.ndarray, last_add, FstarProp):
    '''
    尝试检测已加入的变量中是否有需要删除的。用于逐步回归分析
    :param R: 相关系数矩阵
    :param sample_num: 样本数目n
    :param var_list: 记录各个变量是否加入[bool,...]，以bool方式记录
    :param last_add: 上次加入的变量下标
    :param FstarProp: F检验临界概率
    :return: 成功删除变量为True，没有需要删除为False
    '''
    if last_add is None:
        return False
    var_index = None        # 可能需要剔除的变量下标
    partial_reg = None      # 最小的偏回归平方和
    # 遍历找到偏回归平方和最小的
    for i in range(var_list.shape[0]):
        if i == last_add or not var_list[i]:
            continue
        test_partial = R[i, -1] * R[i, -1] / R[i, i]
        if partial_reg is None or partial_reg > test_partial:
            partial_reg = test_partial
            var_index = i
    # F检验并可能去除该变量
    if var_index is not None:
        Ftest = partial_reg / R[-1, -1] * (sample_num - var_list.nonzero()[0].shape[0] - 1)
        # F检验小于临界值
        Fprop = spe.fdtr(1, sample_num - var_list.nonzero()[0].shape[0] - 1, Ftest)
        if Fprop < FstarProp:
            TransFormR(R, var_index)
            var_list[var_index] = False
            return True
    return False


def TryAddVar(R: np.ndarray, sample_num, var_list: np.ndarray, FstarProp):
    '''
    尝试检测加入新的变量。用于逐步回归分析
    :param R: 相关系数矩阵
    :param sample_num: 样本数目n
    :param var_list: 记录各个变量是否加入[bool,...]，以bool方式记录
    :param FstarProp: F检验临界值概率
    :return: 成功加入变量则返回其下标，没有加入返回None
    '''
    var_index = None        # 可能需要剔除的变量下标
    partial_reg = None      # 最小的偏回归平方和
    # 遍历找到偏回归平方和最大的
    for i in range(var_list.shape[0]):
        if var_list[i]:
            continue
        test_partial = R[i, -1] * R[i, -1] / R[i, i]
        if partial_reg is None or partial_reg < test_partial:
            partial_reg = test_partial
            var_index = i
    # F检验并可能加入该变量
    if var_index is not None:
        Ftest = partial_reg / (R[-1, -1] - partial_reg) * (sample_num - var_list.nonzero()[0].shape[0] - 2)
        # F检验大于临界值
        Fprop = spe.fdtr(1, sample_num - var_list.nonzero()[0].shape[0] - 1, Ftest)
        if Fprop > FstarProp:
            TransFormR(R, var_index)
            var_list[var_index] = True
            return var_index
    return None


def LinearAnalyzePOI(cluster_csv, poi_csv, K, min_sum=0):
    '''
    对聚类结果-POI指数进行线性回归分析
    :param cluster_csv: 聚类结果的csv文件或DataFrame，里面是每个场所单元对每类的隶属度
    :param poi_csv: 每个场所的POI统计信息（可以是各类POI数量、相对比例、TF-IDF...）
    :param K: cluster_csv中聚类数，在cluster_csv文件中从cluster1开始
    :param min_sum: 去除POI数量小于min_sum的场所
    '''
    import globalval

    cluster = cluster_csv if isinstance(cluster_csv, pd.DataFrame) else pd.read_csv(cluster_csv)
    pois = pd.read_csv(poi_csv)
    merge = pd.merge(cluster, pois, on='Id')
    if min_sum > 0:
        merge.drop(index=merge[merge['sum'] < min_sum].index, inplace=True)
    # 相关系数阵
    # columns = gdPOI.poi_mytype_en
    columns = globalval.landuse_name_v2
    poiscore = merge[columns]
    clusterscore = merge[['cluster%d' % (i+1) for i in range(K)]]
    corr = np.corrcoef(np.concatenate([clusterscore, poiscore], axis=1), rowvar=False)
    corr = corr[0:K, K:]
    plt.figure(figsize=(10, 6))
    sb.heatmap(corr, cmap=plt.get_cmap('RdBu_r'), annot=True, norm=TwoSlopeNorm(0),
               xticklabels=columns, yticklabels=['cluster%d' % (i+1) for i in range(K)])
    plt.title('correlation coefficients between clusters & POI types')
    # 显著性检验
    Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2) / len(columns)
    signi = spe.fdtr(len(columns), merge.shape[0] - 2, Ftest)
    plt.figure(figsize=(10, 6))
    sb.heatmap(signi, cmap=plt.get_cmap('Reds'), annot=True, fmt='.4g',
               xticklabels=columns, yticklabels=['cluster%d' % (i+1) for i in range(K)])
    plt.title('correlation significance between clusters & POI types')
    plt.show()
    # 逐步回归分析
    for ik in range(1, K + 1):
        x = merge[columns].values
        y = merge['cluster%d' % ik].values.reshape((merge.shape[0], 1))
        print('cluster%d:' % ik)
        StepRegress(np.concatenate([x, y], axis=1), 0.9)
        print('========')


def LinearAnalyzeDist(cluster_csv, poi_csv, odist_csv, ddist_csv,
                      min_sum=0, datanum_stat=None):
    '''
    线性回归分析：TAZ的熵、混合度？、多样性与该TAZ的"平均"OD trip距离的关系
    多样性与不确定性是不一样的
    一些指标公式：
        entropy: 相对Shanon熵 = -sum(p_i * log2(p_i))
        diverse: 相对多样性 = 2 ^ (-sum(p_i * log2(p_i))) = 2 ^ entropy
        abs_entropy: 绝对不确定性，"绝对"在于乘上场所活动强度（用POI密度rou表示） = rou * entropy
        num * diverse: 多样性与强度的结合 = rou * diverse
        abs_diverse: 先求绝对不确定性，而后转多样性(弃用，因为overflow) = 2 ^ (rou * entropy)
    注：用POI密度rou的回归效果比用POI数量N好

    :param cluster_csv: 聚类结果的csv文件或DataFrame，里面是每个场所单元对每类的隶属度
    :param poi_csv: 每个场所的POI统计信息（可以是各类POI数量、相对比例、TF-IDF...）
    :param odist_csv: OD trip距离按照出发地TAZ的平均值统计
    :param ddist_csv: OD trip距离按照目的地TAZ的平均值统计
    :param min_sum: 去除POI数量小于min_sum的场所
    :param datanum_stat: TAZ的OD数、微博数统计，以（可能）去除数据量过少的TAZ
    '''
    # 各变量连接
    cluster = cluster_csv if isinstance(cluster_csv, pd.DataFrame) else pd.read_csv(cluster_csv)
    pois = pd.read_csv(poi_csv) #usecols=['Id', 'sum'])
    columns = list(pois.columns)
    # columns[columns.index('sum')] = 'num'
    pois.columns = columns
    cluster['diverse'] = np.exp2(cluster['entropy'])
    merge = pd.merge(cluster, pois, on='Id')
    # merge['abs_entropy'] = merge['entropy'] * merge['density']
    # merge['abs_diverse'] = merge['diverse'] * merge['density']

    # 测试使用POI点的混合度代替聚类混合度
    # import gdPOI
    # mytype = merge[gdPOI.poi_mytype_en]
    # merge['entropy'] = np.sum(-mytype * np.log2(mytype), axis=1)
    # merge['diverse'] = np.exp2(merge['entropy'])
    # merge['abs_entropy'] = merge['entropy'] * merge['num']
    # merge['abs_diverse'] = merge['diverse'] * merge['num']

    odist = pd.read_csv(odist_csv)
    odist.columns = ['Id', 'odist', 'ocnt']
    ddist = pd.read_csv(ddist_csv)
    ddist.columns = ['Id', 'ddist', 'dcnt']
    merge = pd.merge(merge, odist, on='Id')
    merge = pd.merge(merge, ddist, on='Id')
    # merge.to_csv(path.join(path.dirname(poi_csv), 'v3rand_entropy.csv'), index=False)

    # 可能：去除POI点数，以及OD数、微博数过少的样本
    if min_sum > 0:
        merge.drop(index=merge[merge['num'] < min_sum].index, inplace=True)
    if datanum_stat is not None:
        numstat = pd.read_csv(datanum_stat)
        merge = pd.merge(merge, numstat, on='Id')
        dropid = merge.loc[(merge['o_num'] < 600) | (merge['d_num'] < 600)
                           | (merge['weibo_num'] < 600), 'Id']
        print(merge.shape)
        merge.drop(index=dropid.index, inplace=True)
        print(merge.shape)

    # 计算相关系数
    corr = merge[['entropy', 'diverse', # 'density', 'abs_entropy', 'abs_diverse',
                  'odist', 'ddist']].corr()
    corr = corr.loc[['odist', 'ddist'],
                    ['entropy', 'diverse', ]]# 'density', 'abs_entropy', 'abs_diverse']]
    plt.figure(figsize=(10, 4))
    sb.heatmap(corr, cmap=plt.get_cmap('Blues_r'), annot=True,
               fmt='.4g', annot_kws={'size': 14},
               xticklabels=corr.columns, yticklabels=corr.index)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('correlation coefficients between OD trip distance & TAZ mixing degree', size=14)

    # 显著性检验
    Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2)
    signi = spe.fdtr(1, merge.shape[0] - 2, Ftest)
    plt.figure(figsize=(10, 4))
    sb.heatmap(signi, cmap=plt.get_cmap('Reds'), annot=True,
               fmt='.4g', annot_kws={'size': 14},
               xticklabels=corr.columns, yticklabels=corr.index)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('correlation significance between OD trip distance & TAZ mixing degree', size=14)

    # 绘制回归曲线
    for col, oflabel, cnt in (('odist', 'origin TAZ', 'ocnt'), ('ddist', 'destination TAZ', 'dcnt')):
        x_var = 'entropy'
        y = merge[col] / 1000
        plt.figure(figsize=(8, 6))
        sb.regplot(x=merge[x_var], y=y, line_kws={'color': 'black'})
        # plt.scatter(merge[x_var], y)
        lr = LinearRegression()
        lr.fit(merge[x_var].values.reshape((merge.shape[0], 1)), y)
               # sample_weight=merge[cnt])
        # print(np.sqrt(lr.score(merge['abs_diverse'].values.reshape((merge.shape[0], 1)), merge[col],
        #                sample_weight=merge[cnt])))
        # x = np.array([np.min(merge[x_var]), np.max(merge[x_var])])
        # y = lr.predict(x.reshape(x.shape[0], 1))
        # plt.plot(x, y, c='black')
        plt.title('linear regression between entropy and taxi trip distance', size=14)
        plt.xlabel('weighted diversity of ' + oflabel, size=14)
        plt.ylabel('taxi trip distance (km)', size=14)
        plt.xticks(size=14)
        plt.yticks(size=14)
        px = np.max(merge[x_var])
        plt.annotate('y=%.4fx%+.3f, r=%.4f' % (lr.coef_, lr.intercept_, corr.loc[col, x_var]),
                     xy=(px, lr.predict(np.array([[px]]))), xytext=(-300, -20), textcoords='offset pixels', size=14)
    plt.show()


def LinearAnalyzeDist2(cluster_csv, poi_csv, odata_csv, ddata_csv,
                       min_sum=0, datanum_stat=None):
    '''
    线性回归分析：TAZ的熵、混合度？、多样性与OD trip距离的关系（使用"每条"OD记录回归）
    回归效果不理想
    多样性与不确定性是不一样的
    一些指标公式：
        entropy: 相对Shanon熵 = -sum(p_i * log2(p_i))
        diverse: 相对多样性 = 2 ^ (-sum(p_i * log2(p_i))) = 2 ^ entropy
        abs_entropy: 绝对不确定性，"绝对"在于乘上场所活动强度（用POI数量N表示） = N * entropy
        num * diverse: 多样性与强度的结合 = N * diverse

    :param cluster_csv: 聚类结果的csv文件或DataFrame，里面是每个场所单元对每类的隶属度
    :param poi_csv: 每个场所的POI统计信息（可以是各类POI数量、相对比例、TF-IDF...）
    :param odata_csv: 每条OD trip的距离以及其O点的TAZ
    :param ddata_csv: 每条OD trip的距离以及其O点的TAZ
    :param min_sum: 去除POI数量小于min_sum的场所
    :param datanum_stat: TAZ的OD数、微博数统计，以（可能）去除数据量过少的TAZ
    '''
    # 各变量连接
    cluster = cluster_csv if isinstance(cluster_csv, pd.DataFrame) else pd.read_csv(cluster_csv)
    pois = pd.read_csv(poi_csv)  # usecols=['Id', 'sum'])
    columns = list(pois.columns)
    columns[columns.index('sum')] = 'num'
    pois.columns = columns
    cluster['diverse'] = np.exp2(cluster['entropy'])
    merge = pd.merge(cluster, pois, on='Id')
    merge['abs_entropy'] = merge['entropy'] * merge['num']
    merge['abs_diverse'] = merge['diverse'] * merge['num']

    # 可能：去除POI点数，以及OD数、微博数过少的样本
    if min_sum > 0:
        merge.drop(index=merge[merge['num'] < min_sum].index, inplace=True)
    if datanum_stat is not None:
        numstat = pd.read_csv(datanum_stat)
        merge = pd.merge(merge, numstat, on='Id')
        dropid = merge.loc[(merge['o_num'] < 600) | (merge['d_num'] < 600)
                           | (merge['weibo_num'] < 600), 'Id']
        merge.drop(index=dropid.index, inplace=True)

    # 原始OD中路程大于50km的视为异常数据
    odata = pd.read_csv(odata_csv, usecols=['Id', 'dist'])
    odata.columns = ['Id', 'odist']
    odata.drop(index=odata[odata['odist'] > 50000].index, inplace=True)
    ddata = pd.read_csv(ddata_csv, usecols=['Id', 'dist'])
    ddata.columns = ['Id', 'ddist']
    ddata.drop(index=ddata[ddata['ddist'] > 50000].index, inplace=True)
    odata = pd.merge(odata, merge, on='Id')
    ddata = pd.merge(ddata, merge, on='Id')

    # 计算相关系数
    o_corr = odata[['entropy', 'diverse', 'num', 'abs_entropy', 'abs_diverse', 'odist']].corr()
    o_corr = o_corr.loc[['odist'], ['entropy', 'diverse', 'num', 'abs_entropy', 'abs_diverse']]
    d_corr = ddata[['entropy', 'diverse', 'num', 'abs_entropy', 'abs_diverse', 'ddist']].corr()
    d_corr = d_corr.loc[['ddist'], ['entropy', 'diverse', 'num', 'abs_entropy', 'abs_diverse']]
    corr = pd.concat([o_corr, d_corr], axis=0)
    plt.figure(figsize=(10, 4))
    sb.heatmap(corr, cmap=plt.get_cmap('Blues_r'), annot=True,
               fmt='.4g', annot_kws={'size': 14},
               xticklabels=corr.columns, yticklabels=corr.index)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('correlation coefficients between OD trip distance & TAZ mixing degree', size=14)

    # 显著性检验
    Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2)
    signi = spe.fdtr(1, merge.shape[0] - 2, Ftest)
    plt.figure(figsize=(10, 4))
    sb.heatmap(signi, cmap=plt.get_cmap('Reds'), annot=True,
               fmt='.4g', annot_kws={'size': 14},
               xticklabels=corr.columns, yticklabels=corr.index)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title('correlation significance between OD trip distance & TAZ mixing degree', size=14)

    # 绘制回归曲线
    for x, y, oflabel in ((odata['abs_diverse'], odata['odist'], 'origin TAZ'),
                          (ddata['abs_diverse'], ddata['ddist'], 'destination TAZ')):
        plt.figure(figsize=(8, 6))
        plt.hist2d(x, y, bins=100)
        lr = LinearRegression()
        lr.fit(x.values.reshape((x.shape[0], 1)), y)
        x_line = np.arange(np.min(x), np.max(x))
        plt.plot(x_line, lr.predict(x_line.reshape(x_line.shape[0], 1)), c='red')
        plt.title('linear regression between absolute diversity and taxi trip distance', size=14)
        plt.xlabel('absolute diversity of ' + oflabel, size=14)
        plt.ylabel('taxi trip distance', size=14)
        plt.xticks(size=14)
        plt.yticks(size=14)
    plt.show()


def moran_i(shp, column, band_dist=None):
    '''
    全局Moran I指数计算，没有进行行标准化，
    空间权重的定义方式是：面要素重心的欧氏距离平方倒数
    :param shp: shapefile文件路径，或者geopandas.DataFrame
    :param column: 需要计算的变量列（已经存于shp内的列，或者单独的numpy.array）
    :param band_dist: 距离阈值带宽，当两个几何体距离大于band_dist，则空间权重为0
    :return: i, z_score_rand, z_score_norm, p_value_rand, p_value_norm
             分别是原始Moran I值，归一化z值，显著性p。_rand表示随机化假设，_norm表示正态化假设
    '''
    if isinstance(shp, str):
        shp = gpd.read_file(shp)
    if isinstance(column, str):
        column = shp[column]
    if isinstance(column, pd.Series):
        column = column.values
    column = column - np.mean(column)
    assert shp.shape[0] == column.shape[0]
    n = shp.shape[0]
    # 计算空间权重矩阵
    distance = np.zeros((n, n))
    centroid = shp.centroid
    for i in range(n):
        distance[i, :] = centroid.distance(centroid.iloc[i]).values
    distance[distance == 0] = np.inf
    if band_dist is not None:
        distance[distance > band_dist] = np.inf
    weight = 1 / np.square(distance)

    # 计算原始moran I，归一化值，显著性值
    _s0 = np.sum(weight)
    moranI = n * np.sum(weight * (column[:, None] * column[None, :])) \
             / (_s0 * np.sum(np.square(column)))
    Ei = -1 / (n - 1)
    _s1 = np.sum(np.square(weight + weight.T)) / 2
    _s2 = np.sum(np.square(np.sum(weight, axis=0) + np.sum(weight, axis=1)))

    # 随机化假设(randomization assumption)的方差
    _a = n * ((n * n - 3 * n + 3) * _s1 - n * _s2 + 3 * _s0 * _s0)
    _d = np.sum(column ** 4 / n) / np.sum(column ** 2 / n) ** 2
    _b = _d * (n * (n - 1) * _s1 - 2 * n * _s2 + 6 * _s0 * _s0)
    _c = (n - 1) * (n - 2) * (n - 3) * _s0 * _s0
    Vari = (_a - _b) / _c - Ei * Ei

    # 正态假设(normality assumption)的方差
    # Vari_norm = (n * n * (n - 1) * _s1 - n * (n - 1) * _s2 - 2 * _s0 * _s0) \
    #             / ((n + 1) * (n - 1) * _s0 * _s0)
    v_num = n * n * _s1 - n * _s2 + 3 * _s0 * _s0
    v_den = (n - 1) * (n + 1) * _s0 * _s0
    Vari_norm = v_num / v_den - (1 / (n - 1)) ** 2

    z_score_norm = (moranI - Ei) / np.sqrt(Vari_norm)
    z_score = (moranI - Ei) / np.sqrt(Vari)
    p_value = (spe.ndtr(z_score) if z_score < 0 else spe.ndtr(-z_score)) * 2
    p_value_norm = (spe.ndtr(z_score_norm) if z_score_norm < 0 else spe.ndtr(-z_score_norm)) * 2
    return moranI, z_score, z_score_norm, p_value, p_value_norm


def drawCluster(shp, clusters, column=None, title=None, color=None):
    '''
    绘制聚类隶属度结果的空间分布
    :param clusters: 可以是numpy, pandas的对象（聚类结果）或聚类结果csv文件（字符串路径）
    '''
    shpf = gpd.read_file(shp)
    if isinstance(clusters, str):
        clusters = pd.read_csv(clusters)
    if isinstance(clusters, pd.DataFrame):
        clusters = clusters[['Id', column]]
        clusters.columns = ['Id', 'cluster']
        shpf = shpf.merge(clusters, on='Id')
    else:
        shpf['cluster'] = clusters
    shpf.plot(column='cluster', cmap=color, legend=True,
              edgecolor='grey', figsize=(8, 6))
    plt.title(title, size=14)
    plt.axis('off')
    plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)
    plt.show()


if __name__ == '__main__':
    # LinearAnalyzePOI(bishe_root + 'cluster/fcm_k5_v3rand_clip.csv',
    #                  bishe_root + 'analyze/poitaz_tfidfv2.csv', K=5, min_sum=00)
    # LinearAnalyzeDist(bishe_root + 'cluster/fcm_k5_triplet_7x24.csv',
    #                   # bishe_root + 'analyze/poitaz_tfidfv2.csv',
    #                   bishe_root + 'analyze/landuse_stat2.csv',
    #                   bishe_root + 'analyze/odata_dist_stat.csv',
    #                   bishe_root + 'analyze/ddata_dist_stat.csv', min_sum=00)
    #                   # datanum_stat=bishe_root + 'analyze/datanum_stat.csv')
    # LinearAnalyzeDist2(bishe_root + 'fcm_k5_v3rand.csv',
    #                    bishe_root + 'poitaz_tfidfv2.csv',
    #                    bishe_root + 'analyze/odata_dist.csv',
    #                    bishe_root + 'analyze/ddata_dist.csv', min_sum=20,
    #                    datanum_stat=bishe_root + 'datanum_stat.csv')
    shapefile = gpd.read_file(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp')
    stat = pd.read_csv(bishe_root + 'vectors/fcm_k5_triplet_7x24_final.csv')
    shapefile = shapefile.merge(stat, on='Id')
    for item in ['cluster%d' % (d + 1) for d in range(5)] + ['entropy']:
        moranI = moran_i(shapefile, item)
        print(moranI)
    # drawCluster(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
    #             r'E:/各种文档 本四/本科毕设/cluster/fcm_k5_triplet_7x24.csv',
    #             column='entropy', title='entropy', color='RdYlBu_r')
    # drawCluster(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
    #             r'E:/各种文档 本四/本科毕设/cluster/fcm_k5_v3rand.csv',
    #             column='cluster3', title='cluster3', color='Blues')
