import pandas as pd
from os import path
import numpy as np
import re
import time
import geopandas as gpd
import shapely.geometry as geo
import tslearn.metrics as ts
from sklearn.linear_model import LinearRegression


weibo_root = 'E:/gis_data/微博/Sina2016Weibo（处理好的）/'
od_root = 'E:/gis_data/出租车OD/'


def clean_poi_csv1(filename, outputname):
    '''
    清理csv文件的POI
    原始POI共1186983个，范围是北京行政区
    列标题：[ ,name,typecode,wgs84_x,wgs84_y,type1,type2,type3]
    '''
    frame = pd.read_csv(filename)
    frame['wgs84_x'].astype('float64')
    frame['wgs84_y'].astype('float64')

    # 五环经纬度范围：(39.756686, 116.201782) ~ (40.021886, 116.543619)
    ring5frame = frame[(frame['wgs84_y'] > 39.75) & (frame['wgs84_y'] < 40.03) &
                       (frame['wgs84_x'] > 116.20) & (frame['wgs84_x'] < 116.55)].copy()
    column = list(ring5frame.columns)
    column[0] = 'ID'
    ring5frame.columns = column
    ring5frame['ID'] = np.arange(ring5frame.shape[0])
    ring5frame.to_csv(path.join(path.dirname(filename), outputname), index=False)
    # 此时剩余562537个POI


def clean_poi_csv2(filename, shpfile, output, stat_type='ratio'):
    '''
    type1类别： {'住宿服务', '公司企业', '公共设施', '摩托车服务',
        '生活服务', '餐饮服务', '体育休闲服务', '汽车维修', '室内设施',
        '风景名胜', '汽车服务', '医疗保健服务', '汽车销售', '科教文化服务',
        '金融保险服务', '商务住宅', '道路附属设施', '政府机构及社会团体',
        '交通设施服务', '购物服务', '地名地址信息'}
    更细类别见gdPOI分类体系.json

    目标分类：
        {居住: ['住宿服务', {'商务住宅': ['住宅区']}],
        工作教育: ['公司企业', {'生活服务': ['事务所', '中介机构',
            '物流速递'（保留名字带“公司”的）, '人才市场', '信息咨询中心', '旅行社'],
            '科教文化服务': ['科研机构', '培训机构', '学校', '传媒机构'],
            '金融保险服务': ['保险公司', '证券公司', '金融保险服务机构', '财务公司'],
            '商务住宅': ['楼宇', '产业园区']}],
        生活服务: [{'生活服务': ['电力营业厅', '电讯营业厅', '邮局', '摄影冲印店',
            '自来水营业厅', '婴儿服务场所'],
            '金融保险服务': ['银行']}]
        公共服务: ['政府机构及社会团体', {'医疗保健服务': ['动物医疗场所', '疾病预防机构', '诊所', '综合医院',
            '急救中心', '专科医院']}]
        餐饮: ['餐饮服务'],
        购物娱乐: ['购物服务', {'体育休闲服务': ['影剧院', '娱乐场所', '运动场馆'],
            '生活服务': ['洗浴推拿场所', '美容美发店']}],
        观光休闲: ['风景名胜', {'体育休闲服务': ['高尔夫相关', '度假疗养场所', '休闲场所',
            '体育休闲服务场所'],
            '科教文化服务': ['美术馆', '科技馆', '天文馆', '展览馆', '会展中心', '博物馆']}]}
    :param stat_type: 统计类型：'num', 'ratio', 'relative', 'tfidf'中选择一个
        'num'表示统计原始POI数
        对于i场所的j类型POI，'ratio' = N_ij / N_i, 'relative' = (N_ij / N_i) / (N_j / N)
        'tfidf' = tf * idf = (N_ij / N_i) * log(N / N_j)
    '''
    import globalval

    frame = pd.read_csv(filename)

    # 获得高德POI分类体系
    # dicts = {}
    # type1s = set(frame['type1'])
    # for type1 in type1s:
    #     dict1 = {}
    #     frame1 = frame[frame['type1'] == type1]
    #     type2s = set(frame1['type2'])
    #     for type2 in type2s:
    #         frame2 = frame1[frame1['type2'] == type2]
    #         dict1[type2] = list(set(frame2['type3']))
    #     dicts[type1] = dict1
    # print(dicts)

    # 内部详细分析
    def get_ontotype(line):
        '''从poi_ontology中获取该POI类别'''
        if line['type1'] in globalval.poi_ontology:
            layer1 = globalval.poi_ontology[line['type1']]
            if type(layer1) == str:
                return layer1
            if line['type2'] in layer1:
                return layer1[line['type2']]
        return None

    # POI和场所单元空间连接
    frame['mytype'] = frame.apply(get_ontotype, axis=1)
    frame['point'] = frame.apply(generate_p, axis=1, args=('wgs84_y', 'wgs84_x'))
    frame.dropna(inplace=True)
    frame = gpd.GeoDataFrame(frame, geometry='point')
    frame.set_crs('epsg:4326', inplace=True)
    shp = gpd.read_file(shpfile)
    sjoint = gpd.sjoin(frame, shp, op='intersects')
    # 剩下342559个POI
    print('POI num:', sjoint.shape[0])
    cnt = sjoint.groupby(['Id', 'mytype'], as_index=False).count()
    taz_csv = shp.to_crs('epsg:32650')
    taz_csv['area'] = taz_csv.area / 10000
    taz_csv = pd.DataFrame(taz_csv)
    del taz_csv['geometry']
    # save_csv = pd.DataFrame(sjoint[['wgs84_x', 'wgs84_y', 'mytype']])
    # save_csv.to_csv(output + '_poiloc.csv', index=False)

    def get_cnt(line, cnt: pd.DataFrame, mytype):
        '''从cnt中读取统计数据'''
        data = cnt.loc[(cnt['Id'] == line['Id']) & (cnt['mytype'] == mytype), 'name']
        if data.shape[0] == 0:
            data = 0
        else:
            data = data.iloc[0]
        return data

    # 统计各类POI个数和密度（密度单位：个/ha）
    for i in range(len(globalval.poi_mytype)):
        taz_csv[globalval.poi_mytype_en[i]] = taz_csv.apply(get_cnt, axis=1,
                                                            args=(cnt, globalval.poi_mytype[i]))
    taz_csv['sum'] = taz_csv.apply(lambda l: sum([l[s] for s in globalval.poi_mytype_en]), axis=1)
    taz_csv['density'] = taz_csv['sum'] / taz_csv['area']
    sumtype = taz_csv.apply(sum, axis=0)
    # 计算VIF（方差膨胀因子）
    vif = {}
    for i, s in enumerate(globalval.poi_mytype_en):
        lg = LinearRegression()
        others = taz_csv[[item for item in globalval.poi_mytype_en if item != s]]
        lg.fit(others, taz_csv[s])
        vif[s] = 1 / (1 - lg.score(others, taz_csv[s]))
    print('vif:', vif)

    for s in globalval.poi_mytype_en:
        # 计算各地各类别占比
        if stat_type == 'ratio':
            taz_csv[s] = taz_csv[s] / taz_csv['sum']
        # 计算相对量(该地该类别占比 / 全市平均该类别占比)
        elif stat_type == 'relative':
            taz_csv[s] = taz_csv[s] / taz_csv['sum'] / (sumtype[s] / sumtype['sum'])
        # 计算TF-IDF指数
        elif stat_type == 'tfidf':
            taz_csv[s] = taz_csv[s] / taz_csv['sum'] * np.log(sumtype['sum'] / sumtype[s])
        elif stat_type != 'num':
            raise ValueError('stat_type must be one of `num`, `ratio`, `relative`, `tfidf`')

    # 确定各TAZ中指数值最大的POI类型出现在哪类
    taz_csv['major_type'] = taz_csv[globalval.poi_mytype_en].values.argmax(axis=1)
    taz_csv['major_type'] = taz_csv['major_type'].map(lambda v: globalval.poi_mytype_en[v])
    save_csv = taz_csv[['Id', 'major_type']]
    # save_csv.to_csv(output, index=False)

    # 计算shanon熵
    def cal_entrpopy(line):
        entropy = 0
        for s in globalval.poi_mytype_en:
            if line[s] > 0:
                entropy -= line[s] * np.log2(line[s])
        return entropy
    if stat_type == 'ratio':
        taz_csv['entropy'] = taz_csv.apply(cal_entrpopy, axis=1)
    taz_csv.to_csv(output, index=False)


def weibo_textclean1(filename, outputname):
    '''
    微博文本的列有：rowid_for_ARCGIS,weibo_id,weibo_id_by_sina,user_id,weibo_text
    '''

    def weibo_delspetialwords(string):
        '''
        输入一句微博文本，去除特殊字符。需要去除的文本有：
            活动标签或艾特人：格式为#xxx#，@xxx (艾特ID后有空格)
            表情：格式为[xxx]
            发布短链接：格式为http://t.cn/xxxxxxx
            将分隔符（标点，空格，其他字符）统一为空格
        '''
        string1 = re.sub(r'#.*?#', '', string)
        string2 = re.sub(r'@.+ ', '', string1)
        string1 = re.sub(r'\[.*\]', '', string2)
        string2 = re.sub(r'http://t\.cn/.{7}', '', string1)
        string1 = re.sub(u'[^a-z A-Z0-9\u4E00-\u9FA5]', ' ', string2)
        return string1.lower()

    frame = pd.read_csv(filename, usecols=['rowid_for_ARCGIS', 'weibo_id', 'weibo_text'])
    frame['weibo_text'] = frame['weibo_text'].map(weibo_delspetialwords)
    frame.to_csv(path.join(path.dirname(filename), outputname), index=False)


def cal_weekDay(timestamp):
    '''从给定时间戳计算星期几'''
    timestruct = time.localtime(timestamp)
    # weekday的0为星期一，6为星期日
    return timestruct.tm_wday


def cal_hour(timestamp):
    '''从给定时间戳计算小时'''
    timestruct = time.localtime(timestamp)
    return timestruct.tm_hour


def weibo_pointclean1(pointsname, textname, output):
    '''
    整理5ring_points.csv文件
    微博Points的列有：FID,Id,rowid_for_ARCGIS,weibo_id,lon,lat,created_at
    '''

    frame = pd.read_csv(pointsname, usecols=['Id', 'rowid_for_',
                                             'weibo_id', 'lon', 'lat', 'created_at'])
    columns = list(frame.columns)
    columns[0], columns[1] = 'TAZ_ID', 'rowid_for_ARCGIS'
    frame.columns = columns
    frame['weekday'] = frame['created_at'].map(cal_weekDay)
    frame['hour'] = frame['created_at'].map(cal_hour)
    texts = pd.read_csv(textname)
    result = pd.merge(frame, texts, on='weibo_id')
    result.to_csv(path.join(path.dirname(pointsname), output), index=False)


def weibo_fenci(filename, output, stopword=None, strong_mode=False):
    '''
    对微博分词：
    列名：TAZ_ID,rowid_for_ARCGIS_x,weibo_id,lon,lat,created_at,weekday,hour,rowid_for_ARCGIS_y,weibo_text
    用时约30min，剩2402198条数据
    :param stopword: 停用词的文件目录
    :param strong_mode: 表示是否使用强力过滤，True表示仅提取各类名词、动词、形容词
    '''
    import pkuseg

    def fenci(string, model:pkuseg.pkuseg, stwords_):
        return ' '.join(filter(lambda w: w not in stwords_, model.cut(string)))

    frame = pd.read_csv(filename, usecols=['TAZ_ID', 'weibo_id', 'created_at',
                                           'weekday', 'hour', 'weibo_text'])
    model = pkuseg.pkuseg('web')
    frame.dropna(inplace=True)
    if stopword is not None:
        with open(stopword) as file:
            stwords = open(stopword, encoding='utf8').readlines()
            stwords = set(map(lambda s: s[:-1], stwords))
    frame['weibo_fenci'] = frame['weibo_text'].apply(fenci, args=(model, stwords))
    del frame['weibo_text']
    frame.to_csv(path.join(path.dirname(filename), output), index=False)


def weibo_cliptime(filename, timelist, output):
    '''
    根据给定timestamp对应的时间区间，筛选符合要求的微博
    :param timelist: 时间列表，以"%Y-%m-%d %H:%M:%S"格式的[(开始时间1, 结束时间1), (开始2, 结束2), ...]
    选取时间：[('2016-01-04 00:00:00', '2016-02-06 00:00:00'),
              ('2016-02-11 00:00:00', '2016-02-20 00:00:00'),
              ('2016-02-24 00:00:00', '2016-04-04 00:00:00'),
              ('2016-04-06 00:00:00', '2016-04-30 00:00:00'),
              ('2016-05-04 00:00:00', '2016-06-08 00:00:00'),
              ('2016-06-11 00:00:00', '2016-09-14 00:00:00'),
              ('2016-09-17 00:00:00', '2016-09-30 00:00:00'),
              ('2016-10-04 00:00:00', '2016-12-20 00:00:00'),
              ('2016-12-22 00:00:00', '2016-12-24 00:00:00'),
              ('2016-12-27 00:00:00', '2016-12-31 00:00:00')]
    剩余1794622条数据(去除NaN后1764097)
    '''
    frame = pd.read_csv(filename)
    searchlist = []
    for item in timelist:
        start = time.mktime(time.strptime(item[0], "%Y-%m-%d %H:%M:%S"))
        end = time.mktime(time.strptime(item[1], "%Y-%m-%d %H:%M:%S"))
        searchlist.append(frame[(frame['created_at'] >= start) &
                                (frame['created_at'] < end)])
    result = pd.concat(searchlist)
    result.to_csv(path.join(path.dirname(filename), output), index=False)


def OD_clean1(filename, output):
    '''
    从原始出租车轨迹中读取有效的定位点，并转换成OD对

    OD原始数据列名：
        OEM设备类型, 终端ID, 定位发生时间戳, 保留, 纬度*100000, 经度*100000,
        行驶角度, 行驶速度, 行驶里程, 定位描述（0=有效）,
        车辆状态（10000000=载客,0=空车,989681=非运营）, 保留, 车辆状态文字
    车辆轨迹平均采样间隔60s
    某辆车的ID = 00031346d553c683d94d7ab442e4fc71
    '''

    # 读取有效数据：GPS定位有效且处于载客状态
    datalst = []
    with pd.read_csv(filename, sep='\t', chunksize=16 * 1024 * 1024, header=None,
                     usecols=[1, 2, 4, 5, 9, 10]) as reader:
        for chunk in reader:
            # 有时候列内容明明是整数，但pandas还是认为str，需要强制转换
            # 0911开始的异常文件：无
            datalst.append(chunk[(chunk[9] == 0) & (chunk[10] == 10000000)])
    data = pd.concat(datalst)
    print('lines:', data.shape[0])
    data.columns = ['user_id', 'timestamp', 'lat', 'lon', 'gps_valid', 'car_state']
    del data['gps_valid'], data['car_state']
    data.dropna(inplace=True)

    def time_correction(timestamp):
        '''用于时间校正。数据中部分timestamp年份不是2016年'''
        if 1451577600 < timestamp < 1483200000:
            return timestamp
        ts = time.localtime(timestamp)
        return int(time.mktime((2016, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min,
                                ts.tm_sec, 0, 0, 0)))

    def generate_od(frame: pd.DataFrame):
        '''
        由一辆车的表生成OD对
        '''
        # 2条载客旅途的最小间隔，若2个GPS采样点的时间间隔大于threshold秒数，则视为不同载客旅途
        threshold = 300
        df = pd.DataFrame(columns=['o_time', 'o_lat', 'o_lon', 'd_time', 'd_lat', 'd_lon'])
        i_start, i_end = 0, -1
        temp = frame.sort_values('timestamp')
        # 行循环，不断生成OD对
        for i in range(temp.shape[0]):
            if i == i_end + 1:
                i_start = i
            if i + 1 == temp.shape[0] or \
                    temp.iloc[i+1, 1] - temp.iloc[i, 1] > threshold:
                i_end = i
                if i_end > i_start:
                    df = df.append({'o_time': temp.iloc[i_start, 1], 'o_lat': temp.iloc[i_start, 2],
                                    'o_lon': temp.iloc[i_start, 3], 'd_time': temp.iloc[i_end, 1],
                                    'd_lat': temp.iloc[i_end, 2], 'd_lon': temp.iloc[i_end, 3]},
                                    ignore_index=True)

        def div10w(x):
            return x / 100000

        # 将原始int表示的经纬度转成浮点数
        df['o_lat'] = df['o_lat'].map(div10w)
        df['o_lon'] = df['o_lon'].map(div10w)
        df['d_lat'] = df['d_lat'].map(div10w)
        df['d_lon'] = df['d_lon'].map(div10w)
        return df

    # 从原始GPS点数据生成OD对
    data['timestamp'] = data['timestamp'].map(time_correction)
    result = data.groupby('user_id').apply(generate_od)
    result.to_csv(output, index=False)
    print('OD finish: lines', result.shape[0])


def generate_p(frame, latname, lonname):
    '''生成一个点'''
    return geo.Point(frame[lonname], frame[latname])


def OD_clean2(odroot, shpfile, output_root):
    '''
    出租车OD表与TAZ的shapefile空间连接，生成O、D的时谱曲线和OD矩阵
    输出od.npy（OD矩阵）、o_timeline.npy, d_timeline.npy（时间序列）
    '''
    shp = gpd.read_file(shpfile)
    days_file = ['201608%02d.csv' % i for i in range(1, 32)] + \
                ['201609%02d.csv' % i for i in range(1, 12)] + \
                ['201609%02d.csv' % i for i in range(19, 26)]
    # od_npy维度: (一周7天（Mon.为下标0）, 一日分12段每段2h, 出发点的场所ID, 目的地的场所ID)
    od_npy = np.zeros((7, 12, shp.shape[0], shp.shape[0]), dtype=int)
    # timeline维度: (场所ID, 时间曲线(1h一段))
    o_timeline = np.zeros((shp.shape[0], 24 * len(days_file)), dtype=int)
    d_timeline = np.zeros((shp.shape[0], 24 * len(days_file)), dtype=int)

    # 每文件（每天）循环
    for i, dayfile in enumerate(days_file):
        # 计算该文件对应日期是周几
        tstamp = time.mktime((int(dayfile[0:4]), int(dayfile[4:6]), int(dayfile[6:8]),
                              0, 0, 0, 0, 0, 0))
        weekday = cal_weekDay(tstamp)

        # 读取文件
        frame = pd.read_csv(odroot + dayfile)
        frame = gpd.GeoDataFrame(frame)
        frame['tripID'] = np.arange(frame.shape[0])
        frame['o_hour'] = frame['o_time'].map(cal_hour)
        frame['d_hour'] = frame['d_time'].map(cal_hour)

        # 生成几何对象
        o_frame = frame.copy()
        o_frame['o_point'] = o_frame.apply(generate_p, axis=1, args=('o_lat', 'o_lon'))
        o_frame.set_geometry('o_point', inplace=True)
        o_frame.set_crs('epsg:4326', inplace=True)
        d_frame = frame.copy()
        d_frame['d_point'] = d_frame.apply(generate_p, axis=1, args=('d_lat', 'd_lon'))
        d_frame.set_geometry('d_point', inplace=True)
        d_frame.set_crs('epsg:4326', inplace=True)
        del o_frame['o_lat'], o_frame['o_lon'], o_frame['d_lat'], o_frame['d_lon']
        del d_frame['o_lat'], d_frame['o_lon'], d_frame['d_lat'], d_frame['d_lon']

        # 空间连接
        # 出发点统计
        ojoin = gpd.sjoin(o_frame, shp, op='intersects')
        ostat = ojoin.groupby(['Id', 'o_hour'], as_index=False).count()
        for line in range(ostat.shape[0]):
            o_timeline[ostat.loc[line, 'Id'], i * 24 + ostat.loc[line, 'o_hour']] = ostat.loc[line, 'tripID']
        # 目的地统计
        djoin = gpd.sjoin(d_frame, shp, op='intersects')
        dstat = djoin.groupby(['Id', 'd_hour'], as_index=False).count()
        for line in range(dstat.shape[0]):
            d_timeline[dstat.loc[line, 'Id'], i * 24 + dstat.loc[line, 'd_hour']] = dstat.loc[line, 'tripID']
        # OD统计
        odjoin = pd.merge(ojoin, djoin, on='tripID')
        # 认为整个trip的时间是出发~到达的时间中点
        odjoin['od_hourbin'] = odjoin.apply(lambda line: cal_hour(
            (line['o_time_x'] + line['d_time_y']) / 2) // 2, axis=1)
        odstat = odjoin.groupby(['Id_x', 'Id_y', 'od_hourbin'], as_index=False).count()
        for line in range(odstat.shape[0]):
            od_npy[weekday, odstat.loc[line, 'od_hourbin'],
                   odstat.loc[line, 'Id_x'], odstat.loc[line, 'Id_y']] = odstat.loc[line, 'tripID']

        print(dayfile, 'finish')

    np.save(output_root + 'od.npy', od_npy)
    np.save(output_root + 'o_timeline.npy', o_timeline)
    np.save(output_root + 'd_timeline.npy', d_timeline)


def timeseries_7x24clean(filename):
    '''
    @deprecated
    将原始时间序列归成7d*24h的一周序列，并标准差归一化(std=1，是每个场所分别归一化)，计算序列间DTW和相似性
    时间序列需先标准差归一化（归一化后|.|^2 = len(时间序列)），然后计算DTW，
        最后基于DTW计算相似性=(|x|^2 + |y|^2 - DTW(x,y)^2) / (2 * |x|^2 * |y|^2)
    '''
    timeline = np.load(filename)
    taz_num = timeline.shape[0]
    t7x24 = np.zeros((taz_num, 24 * 7))
    # 把原始时间序列归并到7*24小时
    for i in range(24 * 7):
        t7x24[:, i] = np.sum(timeline[:, i::(24 * 7)], axis=1)
    del timeline
    # t7x24 = (t7x24 - np.mean(t7x24, axis=1).reshape((taz_num, 1))) / \
    #         np.std(t7x24, axis=1).reshape((taz_num, 1))

    t7x24 = t7x24 / np.std(t7x24, axis=1, keepdims=True)
    dtw = ts.cdist_dtw(t7x24, global_constraint='sakoe_chiba',
                       sakoe_chiba_radius=2, n_jobs=-1)
    meannorm = np.mean(np.linalg.norm(t7x24, axis=1))
    print(meannorm)
    sim = 1 - np.square(dtw) / (2 * (7 * 24 + np.square(meannorm)))
    np.save(path.splitext(filename)[0] + '_7x24_r2.npy', t7x24)
    np.save(path.splitext(filename)[0] + '_dtw_7x24_r2.npy', dtw)
    np.save(path.splitext(filename)[0] + '_sim_7x24_r2.npy', sim)

    print({'max': np.max(dtw), 'min': np.min(dtw), 'mean': np.mean(dtw)})
    print({'max': np.max(sim), 'min': np.min(sim), 'mean': np.mean(sim)})


def timeseries_7x24cleanv2(filename, k=2, norm='abs'):
    '''
    将原始时间序列归成7d*24h的一周序列，归一化，计算序列间DTW和相似性
    absdata为True时，minmax归一化(max=1，全部数据除以统一的max)
        否则标准差归一化（每条序列除以自己的std，归一化后|.|^2 = len(时间序列)），然后计算DTW，
    最后基于DTW计算相似性=(|x|^2 + |y|^2 - DTW(x,y)^2) / (2 * |x|^2 * |y|^2)
        以及计算CORT距离 = 2 / (1 + exp(k * CORT)) * DTW，默认k = 2
        其中CORT = cos相似性(dy / dt)
    :param norm: 归一化参数，支持'abs'(全部场所的数据除以统一的数值),
        'rel'(每个场所除以各自的std), 'log'(时间序列取ln(1+x))
    '''
    timeline = np.load(filename)
    taz_num = timeline.shape[0]
    t7x24 = np.zeros((taz_num, 24 * 7))
    # 把原始时间序列归并到7*24小时
    for i in range(24 * 7):
        t7x24[:, i] = np.sum(timeline[:, i::(24 * 7)], axis=1)
    del timeline
    # 测试使用绝对归一化（保留不同场所的强度比例），即全部场所的数据除以统一的数值
    if norm == 'abs':
        t7x24 = t7x24 / np.max(t7x24)
    # 相对归一化，每个场所分别归一化，分别除以自己的std
    elif norm == 'rel':
        t7x24 = t7x24 / np.std(t7x24, axis=1, keepdims=True)
    else:
        t7x24 = np.log1p(t7x24)
    dtw = ts.cdist_dtw(t7x24, global_constraint='sakoe_chiba',
                       sakoe_chiba_radius=2, n_jobs=-1)
    diffT = np.diff(t7x24, axis=1)
    diffT = diffT / np.linalg.norm(diffT, axis=1, keepdims=True)
    dcort = 2 / (1 + np.exp(k * np.matmul(diffT, diffT.T))) * dtw
    np.save(path.splitext(filename)[0] + '_7x24%s.npy' % norm, t7x24)
    np.save(path.splitext(filename)[0] + '_dtw_7x24%s.npy' % norm, dtw)
    np.save(path.splitext(filename)[0] + '_dcort_7x24%s.npy' % norm, dcort)


def timeseries_2x24clean(filename, k=2, norm='abs'):
    '''
    将原始时间序列归成(周中|周末)*24h的一周序列，并归一化，计算序列间相似性
    时间序列需先标准差归一化（归一化后|.|^2 = len(时间序列)），然后计算DTW，
        基于DTW计算相似性 = (|x|^2 + |y|^2 - DTW(x,y)^2) / (2 * |x|^2 * |y|^2)
        最后计算CORT距离 = 2 / (1 + exp(k * CORT)) * DTW，默认k = 2
        其中CORT = cos相似性(dy / dt)
    :param norm: 归一化参数，支持'abs'(全部场所的数据除以统一的数值),
        'rel'(每个场所除以各自的std), 'log'(时间序列取ln(1+x))
    '''
    timeline = np.load(filename)
    taz_num = timeline.shape[0]
    t2x24 = np.zeros((taz_num, 24 * 2))
    # 把原始时间序列归并到2*24小时
    for i in range(24):
        for j in range(5):
            t2x24[:, i] += np.sum(timeline[:, (24 * j + i)::(24 * 7)], axis=1)
        for j in (5, 6):
            t2x24[:, (i + 24)] += np.sum(timeline[:, (24 * j + i)::(24 * 7)], axis=1)
    t2x24[:, 0:24] = t2x24[:, 0:24] / 5
    t2x24[:, 24:48] = t2x24[:, 24:48] / 2
    del timeline
    # 测试使用绝对归一化（保留不同场所的强度比例），即全部场所的数据除以统一的数值
    if norm == 'abs':
        t2x24 = t2x24 / np.max(t2x24)
    # 相对归一化，每个场所分别归一化，分别除以自己的std
    elif norm == 'rel':
        t2x24 = t2x24 / np.std(t2x24, axis=1, keepdims=True)
    else:
        t2x24 = np.log1p(t2x24)
    dtw = ts.cdist_dtw(t2x24, global_constraint='sakoe_chiba',
                       sakoe_chiba_radius=2, n_jobs=-1)
    diffT = np.diff(t2x24, axis=1)
    diffT = diffT / np.linalg.norm(diffT, axis=1, keepdims=True)
    dcort = 2 / (1 + np.exp(k * np.matmul(diffT, diffT.T))) * dtw
    np.save(path.splitext(filename)[0] + '_2x24%s.npy' % norm, t2x24)
    np.save(path.splitext(filename)[0] + '_dtw_2x24%s.npy' % norm, dtw)
    np.save(path.splitext(filename)[0] + '_dcort_2x24%s.npy' % norm, dcort)

    ratio = 2 / (1 + np.exp(k * np.matmul(diffT, diffT.T)))
    print({'max': np.max(ratio), 'min': np.min(ratio), 'mean': np.mean(ratio)})
    print({'max': np.max(dcort), 'min': np.min(dcort), 'mean': np.mean(dcort)})


def ODdist_stat1(odroot, shpfile, output_root):
    '''
    统计OD的起点、终点所在TAZ，并计算O和D点的距离，以分析路程和TAZ的关系
    本函数从原始经纬度OD读取数据，计算每条OD的路程，
    并输出2个文件（以O点统计TAZ，以D点统计TAZ）
    '''
    shp = gpd.read_file(shpfile)
    days_file = ['201608%02d.csv' % i for i in range(1, 32)] + \
                ['201609%02d.csv' % i for i in range(1, 12)] + \
                ['201609%02d.csv' % i for i in range(19, 26)]
    odatalist = []
    ddatalist = []

    # 每文件（每天）循环
    for dayfile in days_file:
        # 读取文件
        frame = pd.read_csv(odroot + dayfile, usecols=['o_lat', 'o_lon', 'd_lat', 'd_lon'])
        frame = gpd.GeoDataFrame(frame)

        # 生成几何对象
        frame['o_point'] = frame.apply(generate_p, axis=1, args=('o_lat', 'o_lon'))
        frame.set_geometry('o_point', inplace=True)
        frame.set_crs('epsg:4326', inplace=True)
        frame['d_point'] = frame.apply(generate_p, axis=1, args=('d_lat', 'd_lon'))
        frame.set_geometry('d_point', inplace=True)
        frame.set_crs('epsg:4326', inplace=True)

        # 空间连接，计算距离
        for target, tlist in (('o_point', odatalist), ('d_point', ddatalist)):
            frame.set_geometry(target, inplace=True)
            join = gpd.sjoin(frame, shp, op='intersects')
            join.set_geometry('o_point', inplace=True)
            join.to_crs('epsg:32650', inplace=True)
            join.set_geometry('d_point', inplace=True)
            join.to_crs('epsg:32650', inplace=True)
            join['dist'] = join.apply(lambda l: l['o_point'].distance(l['d_point']), axis=1)
            tlist.append(join[['Id', 'dist']])
        print(dayfile, 'finish.')
    odata = pd.concat(odatalist, axis=0, ignore_index=True)
    odata.to_csv(output_root + 'odata_dist.csv', index=False)
    ddata = pd.concat(ddatalist, axis=0, ignore_index=True)
    ddata.to_csv(output_root + 'ddata_dist.csv', index=False)

    # 绘图：OD trip的距离直方图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(odata['dist'] / 1000, bins=500, range=(0, 25), density=True)
    plt.title('histogram of distance of taxi trips', size=14)
    plt.xlabel('trip distance (km)', size=14)
    plt.ylabel('density', size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.show()


def ODdist_stat2(filename, shpfile, output):
    '''
    将OD trip整合到TAZ中，统计平均出行长度
    本函数以~func: `ODdist_stat1`的输出结果作为输入，在TAZ上求平均
    '''
    frame = pd.read_csv(filename)
    # 去除OD距离大于50km的异常数据
    frame.drop(index=frame[frame['dist'] > 50000].index, inplace=True)
    shp = gpd.read_file(shpfile)
    stat_df = pd.DataFrame({'Id': shp['Id'].values, 'dist': np.zeros(shp.shape[0], dtype=float)})
    frame_mean = frame.groupby('Id', as_index=False).mean()
    frame_cnt = frame.groupby('Id', as_index=False).count()

    def get_data(line, mean: pd.DataFrame):
        '''从frame_mean或frame_cnt中读取统计数据'''
        data = mean.loc[mean['Id'] == line['Id'], 'dist']
        if data.shape[0] == 0:
            data = 0
        else:
            data = data.iloc[0]
        return data

    stat_df['dist'] = stat_df.apply(get_data, axis=1, args=(frame_mean,))
    stat_df['cnt'] = stat_df.apply(get_data, axis=1, args=(frame_cnt,))
    stat_df.to_csv(path.join(path.dirname(filename), output), index=False)


def landuse_stat1(landuse_file, taz_file, output):
    '''
    将用地类型统计到taz上
    landuse的类别：住宅、办公、餐饮娱乐、工厂、交通、政府管理、教育、医疗、体育文化中心、公园绿地
    '''
    lu_id = [101, 201, 202, 301, 402, 403, 501, 502, 503, 504, 505]
    from globalval import landuse_name as lu_name
    frame = pd.read_csv(landuse_file, usecols=['Id', 'F_AREA', 'Level2'])
    # 去除无关类型
    # frame = frame[frame['Level2'].isin(lu_id)]
    frame.loc[frame['Level2'] == 403, 'Level2'] = 402
    lu_id.remove(403)
    shp = gpd.read_file(taz_file)
    shp = shp[['Id']]
    cnt = frame.groupby(['Id', 'Level2']).sum()
    total_cnt = frame.groupby('Id').sum()

    def find_area(row, type_id):
        '''统计每条记录（行）的每个类型面积占比'''
        if (row['Id'], type_id) in cnt.index:
            return cnt.loc[(row['Id'], type_id), 'F_AREA'] / total_cnt.loc[row['Id'], 'F_AREA']
        return 0.0

    for i, type_id in enumerate(lu_id):
        shp[lu_name[i]] = shp.apply(find_area, axis=1, args=(type_id,))

    shp.to_csv(path.join(path.dirname(landuse_file), output), index=False)


def landuse_stat2(landuse_file, output):
    '''
    landuse重分类
    分成['residence', 'business', 'shopping', 'industry', 'transport', 'sport_cul', 'park', 'public']
    即把['admin', 'educate', 'medical']3类合成'public'
    '''
    frame = pd.read_csv(landuse_file)
    frame['public'] = frame[['admin', 'educate', 'medical']].sum(axis=1)
    del frame['admin'], frame['educate'], frame['medical']
    frame.to_csv(path.join(path.dirname(landuse_file), output), index=False)


if __name__ == '__main__':
    # POI处理部分
    # clean_poi_csv2('E:/gis_data/POI/19_Beijing5ring_gdPOI.csv',
    #                'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #                'E:/各种文档 本四/本科毕设/poitaz_tfidfv2.csv', stat_type='tfidf')

    # 微博处理部分
    # weibo_textclean1(weibo_root + 'Export_Output_text.csv',
    #                  'weibo_text_step1.csv')
    # weibo_pointclean1(weibo_root + '5ring_points.csv',
    #                   weibo_root + 'weibo_text_step1.csv',
    #                   '5ring_weibo1.csv')
    # weibo_fenci(weibo_root + '5ring_weibo1.csv',
    #             '5ring_weibo_fenci.csv',
    #             stopword=r'E:/gis_data/微博/total_stopwords.txt')
    # weibo_cliptime(weibo_root + '5ring_weibo_fenci.csv',
    #                [('2016-01-04 00:00:00', '2016-02-06 00:00:00'),
    #                 ('2016-02-11 00:00:00', '2016-02-20 00:00:00'),
    #                 ('2016-02-24 00:00:00', '2016-04-04 00:00:00'),
    #                 ('2016-04-06 00:00:00', '2016-04-30 00:00:00'),
    #                 ('2016-05-04 00:00:00', '2016-06-08 00:00:00'),
    #                 ('2016-06-11 00:00:00', '2016-09-14 00:00:00'),
    #                 ('2016-09-17 00:00:00', '2016-09-30 00:00:00'),
    #                 ('2016-10-04 00:00:00', '2016-12-20 00:00:00'),
    #                 ('2016-12-22 00:00:00', '2016-12-24 00:00:00'),
    #                 ('2016-12-27 00:00:00', '2016-12-31 00:00:00')],
    #                '5ring_weibo_result.csv')

    # OD处理部分
    # for i in range(26, 31):
    #     OD_clean1(od_root + ('北京201609/201609%02d.txt' % i),
    #               od_root + ('经纬OD/201609%02d.csv' % i))
    # OD_clean2(od_root + '经纬OD/',
    #           'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #           od_root)
    # timeseries_7x24clean(od_root + 'timeline/o_timeline.npy')
    # timeseries_7x24clean(od_root + 'timeline/d_timeline.npy')
    # timeseries_7x24cleanv2(od_root + 'timeline/o_timeline.npy', norm='log')
    # timeseries_7x24cleanv2(od_root + 'timeline/d_timeline.npy', norm='log')
    # timeseries_2x24clean(od_root + 'timeline/o_timeline.npy', absdata=True)
    # timeseries_2x24clean(od_root + 'timeline/d_timeline.npy', absdata=True)
    # OD trip的距离统计
    # ODdist_stat1(od_root + '经纬OD/',
    #              'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #              'E:/各种文档 本四/本科毕设/analyze/')
    # ODdist_stat2('E:/各种文档 本四/本科毕设/analyze/odata_dist.csv',
    #              'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #              'odata_dist_statv2.csv')
    # ODdist_stat2('E:/各种文档 本四/本科毕设/analyze/ddata_dist.csv',
    #              'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #              'ddata_dist_statv2.csv')

    # 城市用地统计（EULUC）
    # landuse_stat1('E:/各种文档 本四/本科毕设/analyze/lutaz_original.csv',
    #               'E:/gis_data/基础shp/北京交通小区/FiveringTAZ_BJ_wgs84.shp',
    #               'landuse_stat.csv')
    # landuse_stat2('E:/各种文档 本四/本科毕设/analyze/landuse_stat.csv', 'landuse_stat2.csv')
