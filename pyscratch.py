import pandas as pd
import numpy as np
import re
import time
import pkuseg
from os import path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.spatial.distance
from sklearn.manifold import TSNE
import seaborn as sb

import globalval
from cluster import bishe_root

filename = r'E:\gis_data\微博\Sina2016Weibo（处理好的）\Export_Output_points.csv'
vec_dir = r'E:/各种文档 本四/本科毕设/vectors/'


def read_count(filename):
    with open(filename, encoding='utf8') as file:
        i = 0
        line = file.readline()
        while line:
            i += 1
            line = file.readline()
        print('pyline:', i)
    # frame = pd.read_csv(filename)
    # print('pandas line:', frame.shape[0])


def read_sample_lines(filename, num, encoding='utf8', start=0):
    with open(filename, encoding=encoding) as file:
        print(file.readline(4096), end='')
        for i in range(start):
            file.readline(4096)
        for i in range(num):
            print(file.readline(4096), end='')


def read_samplelines_pd(filename, num, sepchar=',', chunk_size=8*1024*1024):
    reader = pd.read_csv(filename, sep=sepchar, chunksize=chunk_size)
    for chunk in reader:
        print(chunk.iloc[0:num])
        break



def timestamp_stat(filename, colname, chunk_size=8*1024*1024):
    chunk = pd.read_csv(filename, chunksize=chunk_size, usecols=[colname,])
    maxtime, mintime = None, None
    for piece in chunk:
        tmax = np.max(piece[colname])
        tmin = np.min(piece[colname])
        if maxtime is None or tmax > maxtime:
            maxtime = tmax
        if mintime is None or tmin < mintime:
            mintime = tmin
    print({'max time': maxtime, 'min time': mintime})


def column_stat(filename, colname, sep='\t', header=None, chunk_size=16*1024*1024):
    '''
    header=None读取无表头文件，默认为0
    chunk_size指分步读取时每步的读取行数
    '''
    reader = pd.read_csv(filename, sep=sep, chunksize=chunk_size, header=header,
                         usecols=[1, 2, 4, 5, 8, 9, 10])
    for i, chunk in enumerate(reader):
        stat = chunk[colname].value_counts()
        stat = chunk[(chunk[9] == 0) & (chunk[10] == 10000000)].shape[0]
        stat = chunk[chunk[8] > 0].shape[0]
        print('chunk %d,' % i, stat)


def group_try(filename):
    reader = pd.read_csv(filename, sep='\t', chunksize=16*1024*1024, header=None,
                         usecols=[1, 2, 4, 5, 9, 10])
    unq = []
    for chunk in reader:
        unq.extend(chunk[1].unique())
    unq = list(set(unq))
    print(len(unq))


def vector_look(filename, group_f=None, title='', metric='euclidean'):
    '''
    查看向量的统计信息、散点图
    '''
    _, extend = path.splitext(filename)
    if isinstance(filename, str):
        if extend == '.npy' or extend == '.npz':
            data = np.load(filename)
        elif extend == '.d2v':
            from gensim.models import doc2vec as d2v
            model = d2v.Doc2Vec.load(filename)
            data = model.dv.vectors
        else:
            return
    else:
        data = filename
    print(data.shape)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    print({'max': np.max(norm), 'min': np.min(norm), 'mean': np.mean(norm)})
    normdata = data / norm
    # 归一化方式：是否需要将散点整体平移至(0, 0)
    # normdata = (normdata - np.mean(normdata, axis=0, keepdims=True)) / np.linalg.norm(normdata, axis=0, keepdims=True)
    # normdata = normdata / np.linalg.norm(normdata, axis=1, keepdims=True)
    sim = np.dot(normdata, normdata.T)
    print({'max': np.max(sim), 'min': np.min(sim), 'mean': np.mean(sim)})

    # 散点图
    tsne = TSNE(metric=metric)
    data2d = tsne.fit_transform(normdata)
    plt.figure(figsize=(8, 6))
    if group_f is None:
        plt.scatter(data2d[:, 0], data2d[:, 1])
    else:
        group = pd.read_csv(group_f)
        from globalval import poi_mytype_en, quxian, quxian_en
        for item in quxian:
            # groupdata = group.loc[group['major_type'] == item, 'Id']
            groupdata = group.loc[group['name'] == item, 'Id']
            plt.scatter(data2d[groupdata.values, 0], data2d[groupdata.values, 1])
        # plt.legend(poi_mytype_en, bbox_to_anchor=(1.01, 0), loc=3, fontsize=14)
        plt.legend(quxian_en, bbox_to_anchor=(1.01, 0), loc=3, fontsize=14)
    plt.title(title, size=14)
    plt.xlabel('x', size=14)
    plt.ylabel('y', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    # 注记
    from globalval import special_taz
    for taz in special_taz:
        plt.annotate(taz[1], xy=(data2d[taz[0], 0], data2d[taz[0], 1]),
                     xytext=(data2d[taz[0], 0] + 0.1, data2d[taz[0], 1] + 0.1), fontname='SimHei')
    plt.subplots_adjust(top=0.95, right=0.75)
    plt.show()


def vector_look3(filenames, group_f=None, metric='cosine', titles=None):
    '''
    查看特征向量的散点图，以多个子图的方式
    '''
    from globalval import poi_mytype_en, quxian, quxian_en, special_taz_en
    fig = plt.figure(figsize=(6.4 * 2, 4.8 * 2), dpi=125)
    grids = gs.GridSpec(4, 4, hspace=0.75, wspace=0.5)
    plt.subplots_adjust(top=0.95, right=0.975, left=0.075, bottom=0.075)
    # fig, axs = plt.subplots(2, 2, figsize=(6.4*len(filenames), 4.8), dpi=125)
    subplot_idx = [(slice(0, 2), slice(0, 2)), (slice(0, 2), slice(2, 4)), (slice(2, 4), slice(1, 3))]
    for i, item in enumerate(filenames):
        ax = plt.subplot(grids[subplot_idx[i]])
        if isinstance(item, str):
            _, extend = path.splitext(item)
            if extend == '.npy' or extend == '.npz':
                data = np.load(item)
            elif extend == '.d2v':
                from gensim.models import doc2vec as d2v
                model = d2v.Doc2Vec.load(item)
                data = model.dv.vectors
            else:
                raise IOError('data not available.', item)
        else:
            data = item

        # 数据降维、绘制
        tsne = TSNE(metric=metric)
        data2d = tsne.fit_transform(data)
        if group_f is None:
            ax.scatter(data2d[:, 0], data2d[:, 1])
        else:
            group = pd.read_csv(group_f) if isinstance(group_f, str) else group_f

            for item in quxian:
                # groupdata = group.loc[group['major_type'] == item, 'Id']
                groupdata = group.loc[group['name'] == item].index
                ax.scatter(data2d[groupdata.values, 0], data2d[groupdata.values, 1])
        # ax.axis('equal')
        ax.set_title(f'({chr(i + 97)}) {titles[i]}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # 注记
        for taz in special_taz_en:
            row_i = group.loc[group['Id'] == taz[0]].index
            ax.annotate(taz[1], xy=(data2d[row_i, 0], data2d[row_i, 1]),
                        xytext=(data2d[row_i, 0] + 0.1, data2d[row_i, 1] + 0.1),
                        bbox={'fc': (1, 1, 1, 0.5), 'lw': 0, 'pad': 0})
    # 图例
    # axs[i].legend(poi_mytype_en, bbox_to_anchor=(1.01, 0), loc=3)
    legend = plt.legend(quxian_en, bbox_to_anchor=(1.03, 0), loc=3)
    plt.text(1.01, 0.45, 'administrative district', ha='left',
             transform=plt.gca().transAxes, fontdict={'fontsize': 11})
    # plt.tight_layout()
    plt.show()
    return fig


def vector_look3_sample():
    '''
    使用vector_look3绘制3个子图的例子（3种特征向量的散点图）
    '''
    from cluster import load_file
    poidf = pd.read_csv(bishe_root + 'analyze/landuse_stat2.csv')
    numstat = pd.read_csv(bishe_root + 'analyze/datanum_stat.csv')
    poidf = pd.merge(poidf, numstat, on='Id')
    # 去除过于稀疏的TAZ
    dropid = poidf.loc[(poidf['o_num'] < 600) | (poidf['d_num'] < 600)
                       | (poidf['weibo_num'] < 600), 'Id']
    del_index = np.array(dropid.index)
    poidf.drop(index=dropid.index, inplace=True)
    vectors = list(load_file(vec_dir + 'od7x24abs_cnn2_triplet_d32_ep100_ml4.npy',
                             [vec_dir + 'justTAZ_d16_len80_num80_p1.0_q0.5.npy',
                              vec_dir + 'justTAZ_d16_len80_num80_p1.0_q2.0.npy'],
                             vec_dir + 'justTAZ_size32_ep40.npy'))
    for i in range(len(vectors)):
        if isinstance(vectors[i], (tuple, list)):
            vectors[i] = np.concatenate(vectors[i], axis=1)
        vectors[i] = np.delete(vectors[i], del_index, axis=0)
    group_f = pd.read_csv(bishe_root + 'analyze/poitaz_quxian.csv')
    group_f = pd.merge(group_f, poidf, on='Id')
    fig = vector_look3(vectors, group_f, titles=['activity dynamic', 'mobility interaction', 'activity semantic'])
    # fig.savefig(bishe_root + 'figure/fig3.pdf')


def matrix_look(filename, title='', xlabel='', scatter=False, group_f=None):
    '''查看相似矩阵的分布直方图、相似散点图'''
    mat = np.load(filename)
    print({'max': np.max(mat), 'min': np.min(mat), 'mean': np.mean(mat),
           'med': np.median(mat), '0.25%': np.quantile(mat, 0.25), '0.75%': np.quantile(mat, 0.75)})
    plt.figure(figsize=(8, 6))
    plt.hist(mat.flatten(), bins=500, density=True)
    plt.title('histogram of ' + title, size=14)
    plt.xlabel(xlabel, size=14)
    plt.ylabel('Density', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.axvline(np.median(mat), ls='--', c='black')   # 竖直辅助线
    plt.axvline(np.quantile(mat, 0.25), ls=':', c='black')  # 0.25分位数的竖直辅助线
    plt.axvline(np.quantile(mat, 0.75), ls=':', c='black')  # 0.25分位数的竖直辅助线
    plt.figure(figsize=(8, 6))

    if scatter:
        # 以输入的相似矩阵为距离，绘制散点图
        tsne = TSNE(metric='precomputed')
        data2d = tsne.fit_transform(mat)
        plt.figure(figsize=(8, 6))
        if group_f is None:
            plt.scatter(data2d[:, 0], data2d[:, 1])
        else:
            group = pd.read_csv(group_f)
            from globalval import poi_mytype_en, quxian, quxian_en
            for item in quxian:
                # groupdata = group.loc[group['major_type'] == item, 'Id']
                groupdata = group.loc[group['name'] == item, 'Id']
                plt.scatter(data2d[groupdata.values, 0], data2d[groupdata.values, 1])
            # plt.legend(poi_mytype_en, bbox_to_anchor=(1.01, 0), loc=3, fontsize=14)
            plt.legend(quxian_en, bbox_to_anchor=(1.01, 0), loc=3, fontsize=14)
        plt.title('scatter of ' + title + ' (by T-SNE)', size=14)
        plt.xlabel('x', size=14)
        plt.ylabel('y', size=14)
        plt.xticks(size=14)
        plt.yticks(size=14)
        # 注记
        from globalval import special_taz
        for taz in special_taz:
            plt.annotate(taz[1], xy=(data2d[taz[0], 0], data2d[taz[0], 1]),
                         xytext=(data2d[taz[0], 0] + 0.1, data2d[taz[0], 1] + 0.1), fontname='SimHei')

    plt.show()


def draw_plot(x, y, title='', xlabel='', ylabel='', legend=None, baseline=None, timetick=None):
    '''绘制折线图'''
    plt.figure(figsize=(8, 6))
    if baseline is None:
        plt.plot(x, y)
    elif type(baseline) in (int, float):
        plt.plot(x, y, x, np.zeros(y.shape) + baseline, '--')
    else:
        plt.plot(x, y, baseline[0], baseline[1], '--')
    plt.title(title, size=14)
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    if legend:
        plt.legend(legend, fontsize=14)
    if timetick:
        plt.xticks(np.arange(x[0], x[-1] + 2, timetick), np.arange(x[0], x[-1] + 2, timetick))
    plt.show()


def draw_errorplot(x, y, minmax, title=None, xlabel=None, ylabel=None, legend=None):
    '''绘制带有误差区间的折线图'''
    fig = plt.figure(dpi=125, figsize=(6.4, 4))
    handles = []
    linestyles = ['-', '--', ':']
    if isinstance(y, (list, tuple)):
        for i in range(len(y)):
            line, = plt.plot(x, y[i], linestyle=linestyles[i])
            interval = plt.fill_between(x, minmax[i][0], minmax[i][1], alpha=0.2)
            handles.append((line, interval))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x[::3], x[::3])
    # y2ord = np.diff(y, n=2)
    # deriv, = plt.plot(x[1:-1], y2ord, label=legend[1] if legend else None)
    if legend is not None:
        plt.legend(handles, legend)
    plt.grid(linestyle=':')
    plt.show()
    return fig


def draw_fpc_silhouette_curve(fpc_file, silhouette_file):
    '''绘制多次运行后的FPC曲线平均值'''
    fpcs = pd.read_csv(fpc_file).values
    mean_fpcs = np.mean(fpcs, axis=0)
    minmax_fpc = [np.max(fpcs, axis=0), np.min(fpcs, axis=0)]
    silhouette = pd.read_csv(silhouette_file).values
    mean_sil = np.mean(silhouette, axis=0)
    minmax_sil = [np.max(silhouette, axis=0), np.min(silhouette, axis=0)]
    fig = draw_errorplot(np.arange(mean_fpcs.shape[0]) + 2, [mean_fpcs, mean_sil],
                         [minmax_fpc, minmax_sil], xlabel='number of clusters $K$',
                         ylabel='values', legend=['fuzzy partition coefficient', 'silhouette score'])
    # fig.savefig(bishe_root + 'figure/fig4.pdf')


def draw_boxplot(filename, title='', ylabel=''):
    '''绘制POI数量在各类的箱线图'''
    from globalval import poi_mytype_en
    frame = pd.read_csv(filename)
    plt.figure(figsize=(9, 6))
    sb.boxplot(data=frame[poi_mytype_en])
    plt.title(title, size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.ylabel(ylabel, size=14)
    plt.show()


def draw_scatter(x, y, dense_plot, title='', xlabel='', ylabel=''):
    '''绘制x, y散点图
    :param dense_plot: True表示用密度直方图代替散点图（用于散点数量过多的情形）'''
    plt.figure(figsize=(8, 6))
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x.reshape((x.shape[0], 1)), y)
    print(lr.score(x.reshape((x.shape[0], 1)), y))
    if not dense_plot:
        sb.regplot(x=x, y=y, scatter_kws={'alpha': 0.01, 's': 3},
                   line_kws={'color': 'black'})
    else:
        plt.hist2d(x, y, bins=100)
        x_line = np.array([np.min(x), np.max(x)])
        plt.plot(x_line, lr.predict(x_line.reshape(x_line.shape[0], 1)), c='red')
    plt.title(title, size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.show()


def draw_sim_map(vector, shpfile, basic_id=538, title=''):
    '''绘制向量相似性地图，
    :param basic_id: 基准TAZ，设置颜色与基准TAZ的相似度有关'''
    import geopandas as gpd
    shp = gpd.read_file(shpfile)
    data = vector / np.linalg.norm(vector, axis=1, keepdims=True)
    sim_mat = np.matmul(data, data.T)
    df = pd.DataFrame({'Id': np.arange(vector.shape[0]), 'similarity_': sim_mat[basic_id, :]})
    shp = shp.merge(df, on='Id')
    shp.to_crs('epsg:32650', inplace=True)
    shp.plot(column='similarity_', cmap='RdYlBu_r', legend=True,
             edgecolor='grey', figsize=(8, 6))
    plt.title(title, size=14)
    plt.axis('off')
    plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)
    plt.show()


def basic_stat(o_statf, d_statf, weibo_pf, poistat, output):
    '''
    基础信息在TAZ上统计：OD量、微博量
    :param o_stat: O点统计的OD
    :param d_stat: D点统计的OD
    :param weibo_p: 微博-坐标对应
    '''
    ofile = pd.read_csv(o_statf)
    ostat = ofile.groupby('Id', as_index=False).count()
    dfile = pd.read_csv(d_statf)
    dstat = dfile.groupby('Id', as_index=False).count()
    weibof = pd.read_csv(weibo_pf)
    weibostat = weibof.groupby('Id', as_index=False).count()
    tazs = pd.read_csv(poistat, usecols=['Id', 'sum'])
    tazs.columns = ['Id', 'poi_num']

    def get_cnt(line, cnt: pd.DataFrame, col):
        '''从cnt中读取统计数据'''
        data = cnt.loc[cnt['Id'] == line['Id'], col]
        return 0 if data.shape[0] == 0 else data.iloc[0]

    tazs['o_num'] = tazs.apply(get_cnt, axis=1, args=(ostat, 'dist'))
    tazs['d_num'] = tazs.apply(get_cnt, axis=1, args=(dstat, 'dist'))
    tazs['weibo_num'] = tazs.apply(get_cnt, axis=1, args=(weibostat, 'weibo_id'))
    tazs.to_csv(output, index=False)
    # tazs = pd.read_csv(output)

    # 绘图
    data = tazs['poi_num']
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=100)
    plt.title('histogram of POIs grouped by TAZ', size=14)
    plt.xlabel('num of POIs', size=14)
    plt.ylabel('frequency', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.axvline(np.median(data), ls='--', c='black')   # 竖直辅助线
    plt.axvline(np.quantile(data, 0.25), ls=':', c='black')  # 0.25分位数的竖直辅助线
    plt.axvline(np.quantile(data, 0.75), ls=':', c='black')  # 0.25分位数的竖直辅助线
    plt.show()
    # print({'max': np.max(data), 'min': np.min(data),
    #        'mean': np.mean(data), 'median': np.median(data), 'std': np.std(data)})
    # plt.figure(figsize=(8, 6))
    # sb.boxplot(data=tazs[['o_num', 'd_num']])
    # plt.xticks(size=14)
    # plt.yticks(size=14)
    # plt.show()


def draw_map(shp, stat_file, column=None, ax=None, title=None, color=None, cbar_title=None,
             force_cat=False, cluster_names=None, cbar_title_pos=None, legend_kw=None):
    '''
    绘制空间分布地图
    :param stat_file: 可以是numpy, pandas的对象（聚类结果）或聚类结果csv文件（字符串路径）
    '''
    import geopandas as gpd
    shpf = gpd.read_file(shp)
    if isinstance(stat_file, str):
        stat_file = pd.read_csv(stat_file)
    if isinstance(stat_file, pd.DataFrame):
        stat_file = stat_file[['Id', column]]
        stat_file.columns = ['Id', 'cluster']
        shpf = shpf.merge(stat_file, on='Id')
    else:
        shpf['cluster'] = stat_file
    ax.set_title(title)
    if force_cat:
        shpf['cluster'] = shpf['cluster'].map({i+1: n for i, n in enumerate(cluster_names)})
        shpf.plot(column='cluster', cmap=color, legend=True, edgecolor='grey',
                  ax=ax, categorical=True, categories=cluster_names,
                  legend_kwds={'loc': (0.8, -0.05)})
        bbox = ax.get_position()
        bbox.x1 = (bbox.x1 - bbox.x0) * 0.9 + bbox.x0
        ax.set_position(bbox)
    else:
        if legend_kw is None:
            legend_kw = {}
        legend_kw.setdefault('shrink', 0.8)
        shpf.plot(column='cluster', cmap=color, legend=True, edgecolor='grey',
                  ax=ax, legend_kwds=legend_kw)
    if cbar_title is not None:
        if cbar_title_pos is None:
            cbar_title_pos = {}
        cbar_title_pos.setdefault('x', 1.2)
        cbar_title_pos.setdefault('y', 1.0)
        ax.text(cbar_title_pos['x'], cbar_title_pos['y'],
                cbar_title, ha='right', transform=ax.transAxes)
    ax.set_axis_off()


def draw_function_proportion(shp, stat_file):
    '''以子图形式，绘制5个城市功能的空间分布比例，以及最高占比城市功能的空间分布'''
    from matplotlib import colors, cm
    fig, axs = plt.subplots(2, 3, figsize=(6.4 * 2.4, 4.8 * 2), dpi=125)
    plt.subplots_adjust(top=0.98, right=0.98, left=0.02, bottom=0.02, wspace=0.05, hspace=0.05)
    colormap = ['Reds', 'Oranges', 'Greens', 'Blues', 'Purples', 'Greys']
    for i in range(5):
        draw_map(shp, stat_file, column=f'cluster{i+1}', cbar_title='proportion',
                 title=f'({chr(i + 97)}) {globalval.mixuse_name[i]}', color=colormap[i], ax=axs.flatten()[i])
    # 制作离散cmap，绘制最高占比城市功能的空间分布
    lst_cm = colors.ListedColormap([cm.get_cmap(item)(value) for item, value in
                                    zip(colormap[:5], [0.6, 0.4, 0.4, 0.4, 0.6])])
    draw_map(shp, stat_file, column='main_cluster', force_cat=True, cluster_names=globalval.mixuse_name,
             title='(f) dominant urban function', color=lst_cm, ax=axs.flatten()[5])
    plt.show()
    fig.savefig(bishe_root + 'figure/fig6.pdf')


def draw_correlation_sample(cluster_csv, landuse_csv, K=5):
    '''绘制聚类结果与土地利用的相关性矩阵'''
    import scipy.special as spe

    cluster = cluster_csv if isinstance(cluster_csv, pd.DataFrame) else pd.read_csv(cluster_csv)
    landuse = pd.read_csv(landuse_csv)
    merge = pd.merge(cluster, landuse, on='Id')
    # 相关系数阵
    # columns = gdPOI.poi_mytype_en
    columns = globalval.landuse_name_v2
    poiscore = merge[columns]
    clusterscore = merge[['cluster%d' % (i + 1) for i in range(K)]]
    corr = np.corrcoef(np.concatenate([clusterscore, poiscore], axis=1), rowvar=False)
    corr = corr[0:K, K:]
    # 显著性检验
    Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2) / len(columns)
    p_value = spe.fdtr(len(columns), merge.shape[0] - 2, Ftest)
    annotation = np.array([f'{v:.3f}' for v in corr.flatten()]).reshape(corr.shape)
    for _p in (0.9, 0.99, 0.999):
        annotation = np.where(p_value > _p, np.char.add(annotation, '*'), annotation)
    # 绘图
    fig = plt.figure(figsize=(8, 6), dpi=125)
    plt.subplots_adjust(top=0.9, right=0.95, left=0.08, bottom=0.12)
    xticklabel = list(columns)
    xticklabel[5] = 'sport &\ncultural'
    ax = sb.heatmap(corr, cmap=plt.get_cmap('RdBu_r'), center=0, annot=annotation, fmt='s',
                    xticklabels=xticklabel, yticklabels=globalval.mixuse_name, cbar_kws={'fraction': 0.05})
    ax.set_xlabel('land use')
    ax.set_ylabel('estimated mixed urban functions')
    ax.text(1.02, 1.03, 'correlation\ncoefficient', ha='left', transform=ax.transAxes)
    plt.show()
    fig.savefig(bishe_root + 'figure/fig5.pdf')


def draw_multi_correlation(cluster_csvs, landuse_csv, names, K=5):
    '''绘制聚类结果与土地利用的相关性矩阵：多子图形式'''
    from mpl_toolkits.axes_grid1 import ImageGrid
    import scipy.special as spe

    fig = plt.figure(figsize=(12, 6.4), dpi=125)
    axs = ImageGrid(fig, 111, (1, len(cluster_csvs)), share_all=True,
                    cbar_location='right', cbar_mode="single", cbar_size='7%',
                    axes_pad=0.2)
    # fig, axs = plt.subplots(1, 3, figsize=(6.4*3, 4.8), dpi=125, sharey=True,
    #                         cbar_location='right', cbar_mode="single")
    # plt.subplots_adjust(top=0.9, right=0.93, left=0.12, bottom=0.12)
    landuse = pd.read_csv(landuse_csv)

    data, pvalues = [], []
    # columns = gdPOI.poi_mytype_en
    columns = globalval.landuse_name_v2
    for i, cluster_csv in enumerate(cluster_csvs):
        cluster = cluster_csv if isinstance(cluster_csv, pd.DataFrame) else pd.read_csv(cluster_csv)
        merge = pd.merge(cluster, landuse, on='Id')
        # 相关系数阵
        poiscore = merge[columns]
        clusterscore = merge[['cluster%d' % (i + 1) for i in range(K)]]
        corr = np.corrcoef(np.concatenate([clusterscore, poiscore], axis=1), rowvar=False)
        corr = corr[0:K, K:]
        data.append(corr.T)
        # 显著性检验
        Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2) / len(columns)
        p_value = spe.fdtr(len(columns), merge.shape[0] - 2, Ftest)
        pvalues.append(p_value.T)
    minmax = [min(map(np.min, data)), max(map(np.max, data))]
    # 绘图
    yticklabel = list(columns)
    yticklabel[5] = 'sport &\ncultural'
    for i, corr in enumerate(data):
        ax = axs[i]
        annot = np.array([f'{v:.2f}' for v in corr.flatten()]).reshape(corr.shape)
        for _p in (0.9, 0.99, 0.999):
            annot = np.where(pvalues[i] > _p, np.char.add(annot, '*'), annot)
        annot = np.where((np.abs(corr) > 0.2) | (corr == np.max(corr, axis=0, keepdims=True)) |
                         (corr == np.min(corr, axis=0, keepdims=True)), annot, '')
        sb.heatmap(corr, cmap=plt.get_cmap('RdBu_r'), cbar_ax=ax.cax, annot=annot, fmt='s',
                   vmin=minmax[0], vmax=minmax[1], center=0,
                   xticklabels=[f'c{_k}' for _k in range(1, K + 1)], yticklabels=yticklabel, ax=ax)
        ax.set_title(f'({chr(i + 97)}) {names[i]}')
    fig.supxlabel('urban function clusters')
    fig.supylabel('land use')
    axs[-1].text(1.02, 1.03, 'correlation\ncoefficient', ha='left', transform=axs[-1].transAxes)
    plt.tight_layout()
    plt.show()
    fig.savefig(bishe_root + 'figure/fig7.pdf')


def draw_lenear_reg(stat_file, x='entropy', y=None, ax=None, title=None, ylabel=None):
    '''
    绘制线性回归的散点、回归直线和置信区间
    '''
    import scipy.special as spe
    from sklearn.linear_model import LinearRegression

    merge = stat_file if isinstance(stat_file, pd.DataFrame) else pd.read_csv(stat_file)
    # 计算相关系数，显著性检验
    corr = merge[[x, y]].corr().values[0, 1]
    Ftest = np.square(corr) / (1 - np.square(corr)) * (merge.shape[0] - 2)
    significance = max(1 - spe.fdtr(1, merge.shape[0] - 2, Ftest), 0.001)

    # 绘制回归曲线
    y = merge[y] / 1000
    lr = LinearRegression()
    lr.fit(merge['entropy'].values.reshape((merge.shape[0], 1)), y)
    sb.regplot(x=merge['entropy'], y=y, ax=ax,
               line_kws={'color': 'black', 'label': f'$y={lr.coef_[0]:.3f}x+{lr.intercept_:.3f},$\n'
                                                    f'$r={corr:.3f},p<{significance:.3f}$'})
    ax.set_title(title)
    ax.set_xlabel('urban mixture index')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=12)


def draw_mixing_degree(shp_file, stat_file):
    '''
    绘制混合度的空间分布
    '''
    fig = plt.figure(figsize=(6.2, 5), dpi=125)
    plt.subplots_adjust(top=1, bottom=0.05, left=0, right=1)
    draw_map(shp_file, stat_file, column='entropy', ax=plt.gca(),
             title=None, color='RdYlBu_r', cbar_title='mixture index',
             cbar_title_pos={'x': 1.2, 'y': 0.93}, legend_kw={'aspect': 30})
    plt.title('Moran\'s $I=0.164$, $p<0.001$', fontsize=11, y=0, pad=-6)
    plt.show()
    fig.savefig(bishe_root + 'figure/fig8.pdf')


def draw_mixing_regression(stat_file, odist_csv, ddist_csv):
    '''
    绘制混合度与出行距离的关系（散点图和回归结果）
    '''
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6 * 2, 4.8), dpi=125)
    plt.subplots_adjust(top=0.93, left=0.07, right=0.97, wspace=0.1)
    lr_title = [('odist', 'average taxi trip distance (km)', 'departures (outflows)'),
                ('ddist', None, 'arrivals (inflows)')]
    stat = stat_file if isinstance(stat_file, pd.DataFrame) else pd.read_csv(stat_file)
    odist = pd.read_csv(odist_csv)
    odist.columns = ['Id', 'odist', 'ocnt']
    ddist = pd.read_csv(ddist_csv)
    ddist.columns = ['Id', 'ddist', 'dcnt']
    merge = pd.merge(stat, odist, on='Id')
    merge = pd.merge(merge, ddist, on='Id')
    for i in range(len(lr_title)):
        draw_lenear_reg(merge, y=lr_title[i][0], ax=axs[i],
                        ylabel=lr_title[i][1])
        axs[i].set_title(f'({chr(i + 97)}) {lr_title[i][2]}', fontsize=13)
    plt.show()
    # fig.savefig(bishe_root + 'figure/fig9.pdf')


def draw_mixing_analyze(shp_file, stat_file, odist_csv, ddist_csv):
    '''
    绘制混合度的分析结果，包括混合度的空间分布、与出行距离的散点图与回归结果
    '''
    fig = plt.figure(figsize=(6.4 * 2.25, 4.8 * 2), dpi=125)
    plt.subplots_adjust(top=0.95, bottom=0.075, left=0.025, right=0.95)
    grids = gs.GridSpec(8, 10, hspace=4, wspace=0.5)
    subplot_idx = [(slice(1, 7), slice(0, 6)), (slice(0, 4), slice(6, 10)), (slice(4, 8), slice(6, 10))]
    ax = plt.subplot(grids[subplot_idx[0]])
    draw_map(shp_file, stat_file, column='entropy', ax=ax,
             title='(a)', color='RdYlBu_r', cbar_title='mixture index',
             cbar_title_pos={'x': 1.2, 'y': 0.93}, legend_kw={'aspect': 30})

    # 绘制回归曲线
    lr_title = [('odist', 'trip distance of departure (km)'),
                ('ddist', 'trip distance of arrival (km)')]
    stat = stat_file if isinstance(stat_file, pd.DataFrame) else pd.read_csv(stat_file)
    odist = pd.read_csv(odist_csv)
    odist.columns = ['Id', 'odist', 'ocnt']
    ddist = pd.read_csv(ddist_csv)
    ddist.columns = ['Id', 'ddist', 'dcnt']
    merge = pd.merge(stat, odist, on='Id')
    merge = pd.merge(merge, ddist, on='Id')
    for i in range(len(lr_title)):
        ax = plt.subplot(grids[subplot_idx[i + 1]])
        draw_lenear_reg(merge, y=lr_title[i][0], ax=ax,
                        title=f'({chr(i + 98)})', ylabel=lr_title[i][1])
    plt.show()
    # fig.savefig(bishe_root + 'figure/fig8.pdf')


# read_sample_lines(filename, 200, start=99050)
# np.var()
# read_samplelines_pd(filename, 10, sepchar='\t')
# read_count(filename)
# timestamp_stat(filename, 'created_at')
# column_stat(filename, None)
# group_try(filename)
# result.to_csv('E:/gis_data/出租车OD/hahah.csv', index=False)
# frame = pd.DataFrame({'xxx': [12,12,12,24,24,36,36,36], 'aaa':[1,1,2,1,2,1,2,2],
#                       'bbb': [1,2,3,4,5,6,7,8], 'ccc': np.random.random(8)})
# stat = frame.groupby(['xxx', 'aaa'], as_index=False).count()
# print(stat)
# print(stat.columns)
# days_file = ['201608%02d.csv' % i for i in range(1, 32)] + \
#                 ['201609%02d.csv' % i for i in range(1, 12)] + \
#                 ['201609%02d.csv' % i for i in range(19, 26)]
# for dayf in days_file:
#     tstamp = time.mktime((int(dayf[0:4]), int(dayf[4:6]), int(dayf[6:8]),
#                       0,0,0,0,0,0))
#     print(time.localtime(tstamp).tm_wday)
#
# '''116.44, 39.92, ID = 333'''
# import geopandas as gpd
# import shapely.geometry as geo

# shp = gpd.read_file('E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp')
# sindex = shp['geometry'].sindex
# start = time.time()
# print(list(sindex.intersection(point.bounds)))
# for i in range(1000):
#     point = geo.Point(116.44, 39.92)
#     raw = shp.loc[sindex.intersection(point.bounds)]
#     precise = raw.loc[raw.intersects(point)]
# print(time.time() - start)

# data = np.load(filename)
# print(data.shape)
# proc = np.zeros((data.shape[0], 24 * 7))
# for i in range(24 * 7):
#     proc[:, i] = np.sum(data[:, i::(24*7)], axis=1)
# proc = proc / np.std(proc, axis=1).reshape((proc.shape[0], 1))
# proc = proc / np.max(proc, axis=1).reshape((proc.shape[0], 1))
# argmax = np.argmax(proc)
# print(proc[argmax // (24*7), argmax % (24 * 7)], np.mean(proc), np.mean(np.std(proc, axis=1)))
# print(np.std(proc[argmax // (24*7), :]))
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(proc.shape[1]), proc[argmax // (24*7), :])
# plt.legend(['zhongguancun'])
# plt.show()
# dtw = ts.cdist_dtw(proc, n_jobs=-1)
# selfnorm = ts.cdist_dtw(proc, np.zeros(1), n_jobs=-1)
# selfnorm = np.sum(np.square(proc - np.mean(proc, axis=1).reshape((proc.shape[0], 1))), axis=1)
# print(dtw.shape, selfnorm.shape)
# result = (selfnorm + selfnorm.reshape((1, proc.shape[0])) - np.square(dtw)) / 2
# result = (2 * 7 * 24 - np.square(dtw)) / (2 * 7 * 24)
# print(np.max(result), np.min(result), np.mean(result))

# vec = np.load(r'E:\gis_data\出租车OD\cnn_vec\o7x24_d80_ep10.npy')
# sim = np.load(r'E:\gis_data\出租车OD\o_timeline_sim_7x24.npy')
# vec_norm = np.linalg.norm(vec, axis=1)
# print({'max': np.max(vec_norm), 'min': np.min(vec_norm), 'mean': np.mean(vec_norm)})
# normdata = vec / vec_norm.reshape(vec.shape[0], 1)
# sim_pred = np.dot(normdata, normdata.T)
# differ = np.abs(sim - sim_pred)
# print({'max': np.max(differ), 'min': np.min(differ), 'mean': np.mean(differ),
#        'median': np.median(differ)})

# 绘制折线图
# frame = pd.read_csv(r'E:\各种文档 本四\本科毕设\n2v_d80.csv')
# draw_plot(frame['epoch'].values, frame['loss'].values / 1e6,
#           title='likelihood function in node2vec train epochs',
#           xlabel='iter of epochs', ylabel='likelihood function ($*10^6$)')
# frame = pd.read_csv(r'E:\gis_data\出租车OD\cnn2_7x24abs_triplet_d80_ep100_deep4_ml24.csv')
# draw_plot(frame['epoch'].values, frame['triplet loss'].values,
#           title=r'triplet loss in TCN train epochs ($2\times24$)',
#           xlabel='iter of epochs', ylabel='triplet loss')
# data = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_dtw_2x24.npy')
# argsort = np.argsort(data[503, :])
# print(argsort)

# vector_look(r'E:\gis_data\出租车OD\timeline\o_timeline_7x24abs.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='time series grouped by Origin TAZ (euclidean dist)')
# vector_look(r'E:\gis_data\出租车OD\ts_vec\o7x24_dcort_d48_ep10.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='vectors of time series grouped by Origin TAZ (D_CORT dist)')
# vector_look(r'E:\gis_data\出租车OD\ts_vec\o7x24_r2dist_d64_ep10.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='vectors of time series grouped by Origin TAZ (DTW dist)')
# vector_look(r'E:\gis_data\出租车OD\ts_vec\od7x24abs_cnn_triplet_d32_ep100_ml4.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='vectors of time series grouped by Origin TAZ (triplet loss)', metric='cosine')
# vector_look(r'E:\gis_data\出租车OD\timeline\o_timeline_dcort_7x24.npy',
#             title='time series grouped by Origin TAZ (D-CORT dist)',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv')
# vector_look(r'E:\gis_data\出租车OD\timeline\o_timeline_dtw_7x24_r2.npy',
#             title='time series grouped by Origin TAZ (DTW dist)',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv')
# vector_look3_sample()
# draw_fpc_silhouette_curve(vec_dir + 'fpc_values.csv', vec_dir + 'silscore_values.csv')
draw_correlation_sample(vec_dir + 'fcm_k5_triplet_7x24_final.csv',
                        r'E:/各种文档 本四/本科毕设/analyze/landuse_stat2.csv')
# draw_function_proportion(r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
#                          vec_dir + 'fcm_k5_triplet_7x24_final.csv')
# draw_multi_correlation([vec_dir + f'fcm_k5_ablation_{i}.csv' for i in range(3)],
#                        r'E:/各种文档 本四/本科毕设/analyze/landuse_stat2.csv',
#                        ['activity dynamic', 'mobility interaction', 'activity semantic'])
# draw_mixing_analyze(bishe_root + r'交通小区/FiveringTAZ_BJ.shp',
#                     vec_dir + 'fcm_k5_triplet_7x24_final.csv',
#                     bishe_root + 'analyze/odata_dist_stat.csv',
#                     bishe_root + 'analyze/ddata_dist_stat.csv')
# draw_mixing_degree(bishe_root + r'交通小区/FiveringTAZ_BJ.shp',
#                    vec_dir + 'fcm_k5_triplet_7x24_final.csv')
# draw_mixing_regression(vec_dir + 'fcm_k5_triplet_7x24_final.csv',
#                        bishe_root + 'analyze/odata_dist_stat.csv',
#                        bishe_root + 'analyze/ddata_dist_stat.csv')


# vector_look(r'E:\gis_data\微博\Sina2016Weibo（处理好的）\doc2vec_vectors\justTAZdbow_size32_ep40.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='vectors of TAZs in weibo doc2vec (by t-SNE)', metric='cosine')
# vector_look(r'E:\gis_data\出租车OD\node2vec\版本2\justTAZ_d32_len80_num80_p1.0_q2.0.npy',
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv',
#             title='vectors of TAZs in OD trip node2vec @p=1,q=2 (by t-SNE)', metric='cosine')

# t7x24 = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_7x24_r2.npy')
# n = t7x24.shape[0]
# normeu = np.linalg.norm(t7x24, axis=1)
# diff = t7x24.reshape((n, 1, t7x24.shape[1])) - t7x24.reshape((1, n, t7x24.shape[1]))
#
# dtw = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_dtw_7x24_r2.npy')
# norm2 = np.square(np.linalg.norm(t7x24, axis=1))
# sim = (norm2.reshape((t7x24.shape[0], 1)) +
#        norm2.reshape((1, t7x24.shape[0])) - np.square(dtw)) / \
#         (2 * np.sqrt(norm2.reshape((t7x24.shape[0], 1))) *
#          np.sqrt(norm2.reshape((1, t7x24.shape[0]))))
# print({'max': np.max(dtw), 'min': np.min(dtw), 'mean': np.mean(dtw)})
# print({'max': np.max(sim), 'min': np.min(sim), 'mean': np.mean(sim)})
# argmax = np.argmax(sim)
# i, j = argmax // t7x24.shape[0], argmax % t7x24.shape[0]
# print(np.sqrt(norm2[i]), np.sqrt(norm2[j]))
# print(dtw[i, j])
# print(np.sum(np.square(t7x24[i, :] - t7x24[j, :])))
# x = np.arange(7*24)
# plt.plot(x, t7x24[i, :], x, t7x24[j, :])
# for i in range(5):
#     plt.figure()
#     rnd = np.random.randint(0, t7x24.shape[0] - 1, size=(2,))
#     plt.plot(x, t7x24[rnd[0], :], x, t7x24[rnd[1], :])
#     print(dtw[rnd[0], rnd[1]])
# plt.show()
# matrix_look(r'E:\gis_data\出租车OD\timeline\o_timeline_dcort_2x24.npy',
#             title='D_CORT distance', xlabel='D_CORT', scatter=True,
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv')
# matrix_look(r'E:\gis_data\出租车OD\timeline\o_timeline_dtw_2x24.npy',
#             title='DTW distance', xlabel='D_CORT', scatter=True,
#             group_f=r'E:\各种文档 本四\本科毕设\analyze\poitaz_quxian.csv')

# basic_stat(r'E:\各种文档 本四\本科毕设\analyze\odata_dist.csv',
#            r'E:\各种文档 本四\本科毕设\analyze\ddata_dist.csv',
#            r'E:\gis_data\微博\Sina2016Weibo（处理好的）\5ring_points.csv',
#            r'E:\各种文档 本四\本科毕设\poitaz_tfidfv2.csv',
#            r'E:\各种文档 本四\本科毕设\datanum_stat.csv')

# 各类POI数目的箱线图
# draw_boxplot(r'E:\各种文档 本四\本科毕设\poitaz_numv2.csv',
#              title='box plot of POIs num in categories', ylabel='num of POIs')

# 绘制北大的时间曲线
# odata = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline.npy')
# ddata = np.load(r'E:\gis_data\出租车OD\timeline\d_timeline.npy')
# data = [odata, ddata]
# for id in range(2):
#     timeline = data[id]
#     taz_num = timeline.shape[0]
    # 归并至7x24
    # t7x24 = np.zeros((taz_num, 24 * 7))
    # for i in range(24 * 7):
    #     t7x24[:, i] = np.sum(timeline[:, i::(24 * 7)], axis=1)
    # data[id] = t7x24
    # 归并至2x24
    # t2x24 = np.zeros((taz_num, 24 * 2))
    # for i in range(24):
    #     for j in range(5):
    #         t2x24[:, i] += np.sum(timeline[:, (24 * j + i)::(24 * 7)], axis=1)
    #     for j in (5, 6):
    #         t2x24[:, (i + 24)] += np.sum(timeline[:, (24 * j + i)::(24 * 7)], axis=1)
    # t2x24[:, 0:24] = t2x24[:, 0:24] / 5
    # t2x24[:, 24:48] = t2x24[:, 24:48] / 2
    # data[id] = t2x24
# odata, ddata = data
# draw_plot(np.arange(odata.shape[1]), odata[203, :],
#           baseline=(np.arange(odata.shape[1]), ddata[203, :]),
#           legend=['origin', 'destination'],
#           title='time series of of OD num in TAZ ID=203',
#           xlabel='hours (weekday at front)', ylabel='num of OD', timetick=6)
# draw_plot(np.arange(odata.shape[1]), odata[40, :],
#           title='time series of of OD num in TAZ ID=40 (as origin)',
#           xlabel='hours from 0:00 Mon.', ylabel='num of OD', timetick=24)
# print(np.max(odata), np.min(odata))
# print(np.max(ddata), np.min(ddata))
# draw_plot(np.arange(ddata.shape[1]), ddata[538, :],
#           title='time series of of OD num in TAZ ID=538 (as destination)',
#           xlabel='hours from 0:00 Mon.', ylabel='num of OD', timetick=24)

# 绘制triplet相似性、dtw距离、...计算得到的相似性散点图，
# 用于分析不同相似性指标是否等价
# triplet = np.load(r'E:\gis_data\出租车OD\ts_vec\od2x24abs_triplet_d80_ep200_ml24.npy')
# triplet = triplet / np.linalg.norm(triplet, axis=1, keepdims=True)
# timeline = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_7x24.npy')
# dtw_dist = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_dtw_7x24_r2.npy')
# dcort_dist = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_dcort_7x24.npy')
# triplet_dist = np.matmul(triplet, triplet.T)
# l2_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(timeline))
# taz_num = triplet.shape[0]
# index = [i for i in range(taz_num * taz_num) if i // taz_num != i % taz_num]
# draw_scatter(triplet_dist.flatten()[index], l2_dist.flatten()[index], False,
#              title='scatter plot of `cosine similarity of triplet embeddings` and `euclidean distance`',
#              xlabel='cosine similarity of triplet embeddings', ylabel='euclidean distance')
# draw_scatter(triplet_dist.flatten()[index], dtw_dist.flatten()[index], False,
#              title='scatter plot of `cosine similarity of triplet embeddings` and `dtw distance`',
#              xlabel='cosine similarity of triplet embeddings', ylabel='dtw distance')
# draw_scatter(triplet_dist.flatten()[index], dcort_dist.flatten()[index], False,
#              title='scatter plot of `cosine similarity of triplet embeddings` and `dcort distance`',
#              xlabel='cosine similarity of triplet embeddings', ylabel='dcort distance')

# 相似性地图
# triplet = np.load(r'E:\gis_data\出租车OD\ts_vec\od7x24abs_triplet_d80_ep200_ml24.npy')
# plt.rcParams['font.sans-serif'] = ['simhei']
# plt.rcParams['axes.unicode_minus'] = False
# for place in gdPOI.special_taz:
#     draw_sim_map(triplet, shpfile=r'E:/各种文档 本四/本科毕设/交通小区/FiveringTAZ_BJ.shp',
#                  title=f'similarity map of triplet vectors with {place[1]}', basic_id=place[0])

# otl = np.load(r'E:\gis_data\出租车OD\timeline\o_timeline_7x24.npy')
# x = np.arange(otl.shape[1])
# x = x.reshape(1, *x.shape)
# draw_plot(x.repeat(3, axis=0).T, otl[[538, 311, 203], :].T, legend=['pku', 'jinrongjie', 'tiantan'])
# print(np.std(otl, axis=1))
