import tensorflow as tf
import tensorflow.keras as kr
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import os


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(5120)])
# 时间曲线的文件夹
file_root = 'E:/gis_data/出租车OD/'


class MyLearnRate(kr.optimizers.schedules.LearningRateSchedule):
    '''
    借鉴于transformer中用的学习率变化公式
    lr = d_model ^ -0.5 * min(step ^ -0.5, step * warmup_steps ^ -1.5)
    但是控制最大learning rate
    一般warm up的比例是0.1，即前10%的epoch是用来warm up的
    '''
    def __init__(self, max_lr, warmup_ep):
        super(MyLearnRate, self).__init__()
        self.max_lr = max_lr
        self.warmup_ep = warmup_ep

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + 1e-3)
        arg2 = step * (self.warmup_ep ** -1.5)
        return tf.math.sqrt(self.warmup_ep) * tf.math.minimum(arg1, arg2) * self.max_lr


class GRUEncoder(kr.Model):
    '''
    基于多层结构GRU的Encoder，可以将一个时间序列转成向量表示
    最后输出可选择是否tanh激活
    '''
    def __init__(self, vec_dim, depth=3, input_expand=False, tanh=True, **kwargs):
        super(GRUEncoder, self).__init__(**kwargs)
        self.input_expand = input_expand
        self.lstms = []
        for i in range(depth - 1):
            self.lstms.append(kr.layers.GRU(vec_dim, recurrent_regularizer='l2', return_sequences=True))
        self.lstms.append(kr.layers.GRU(vec_dim, recurrent_regularizer='l2'))
        activation = 'tanh' if tanh else None
        self.dense = kr.layers.Dense(vec_dim, activation=activation, kernel_regularizer='l2')

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, 2) if self.input_expand else inputs
        for layer in self.lstms:
            x = layer(x, training=training, mask=mask)
        return self.dense(x)


class DilateBlock(kr.layers.Layer):
    # class DilateBlock(kr.Model):
    '''
    用于因果膨胀模型的残差块: 指定filter通道数、膨胀系数dilation
    这个残差块内部2层的膨胀系数【相同】
    '''
    def __init__(self, filters, dilation, need_resample=False, **kwargs):
        '''
        :param filters: 输出向量在特征维的维数
        :param dilation: 膨胀系数，即间隔dilation个时间取一个元素计算
        :param need_resample: 表示输入维数和输出是否不同，需要重采样
        '''
        super(DilateBlock, self).__init__(**kwargs)
        self.conv1 = tfa.layers.WeightNormalization(kr.layers.Conv1D(
            filters, 3, padding='causal', dilation_rate=dilation))
        self.lrelu1 = kr.layers.LeakyReLU(0.01)
        self.conv2 = tfa.layers.WeightNormalization(kr.layers.Conv1D(
            filters, 3, padding='causal', dilation_rate=dilation))
        self.lrelu2 = kr.layers.LeakyReLU(0.01)
        self.res = kr.layers.Conv1D(filters, 1) if need_resample else None

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        res = inputs if self.res is None else self.res(inputs)
        return kr.layers.add([res, x])


class DilateBlock2(kr.layers.Layer):
    # class DilateBlock(kr.Model):
    '''
    用于因果膨胀模型的残差块: 指定filter通道数、膨胀系数dilation
    这个残差块内部2层的膨胀系数【不同】:下一层的膨胀系数是上一层的2倍
    '''
    def __init__(self, filters, dilation, need_resample=False, **kwargs):
        '''
        :param filters: 输出向量在特征维的维数
        :param dilation: 膨胀系数，即间隔dilation个时间取一个元素计算
        :param need_resample: 表示输入维数和输出是否不同，需要重采样
        '''
        super(DilateBlock2, self).__init__(**kwargs)
        self.conv1 = tfa.layers.WeightNormalization(kr.layers.Conv1D(
            filters, 3, padding='causal', dilation_rate=dilation))
        self.lrelu1 = kr.layers.LeakyReLU(0.01)
        self.conv2 = tfa.layers.WeightNormalization(kr.layers.Conv1D(
            filters, 3, padding='causal', dilation_rate=dilation * 2))
        self.lrelu2 = kr.layers.LeakyReLU(0.01)
        self.res = kr.layers.Conv1D(filters, 1) if need_resample else None

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        res = inputs if self.res is None else self.res(inputs)
        return kr.layers.add([res, x])


class DilateEncoder(kr.Model):
    '''
    因果膨胀卷积的Encoder，可以将一个时间序列转成向量表示
    该Encoder是一系列残差块的叠加，残差块的膨胀系数指数增加
    最后输出可选择是否tanh激活
    使用的是`DilatedBlock`
    '''
    def __init__(self, vec_dim, inner_dim, depth=5, input_expand=False, tanh=True, **kwargs):
        '''
        :param vec_dim: 输出的表示向量维度
        :param inner_dim: 内部隐藏层的输出维度
        :param depth: 卷积残差块的数量
        :param tanh: 输出的表示向量是否经过tanh
        '''
        super(DilateEncoder, self).__init__(**kwargs)
        self.input_expand = input_expand
        self.vec_dim = vec_dim
        self.inner_dim = inner_dim
        self.depth = depth
        self.list = [DilateBlock(inner_dim, 1, need_resample=not input_expand)]
        dilation = 2
        for i in range(1, depth - 1):
            self.list.append(DilateBlock(inner_dim, dilation))
            dilation *= 2
        self.list.append(DilateBlock(vec_dim, dilation, need_resample=(vec_dim != inner_dim)))
        self.maxpool = kr.layers.GlobalMaxPool1D()
        activation = 'tanh' if tanh else None
        self.fcn = kr.layers.Dense(vec_dim, activation=activation, kernel_regularizer='l2')

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, 2) if self.input_expand else inputs
        for item in self.list:
            x = item(x, training=training)
        x = self.maxpool(x)
        return self.fcn(x)

    def get_config(self):
        return {'vec_dim': self.vec_dim, 'inner_dim': self.inner_dim,
                'depth': self.depth}


class DilateEncoder2(kr.Model):
    '''
    因果膨胀卷积的Encoder，可以将一个时间序列转成向量表示
    该Encoder是一系列残差块的叠加，残差块的膨胀系数指数增加
    最后输出可选择是否tanh激活
    使用的是`DilatedBlock2`
    '''
    def __init__(self, vec_dim, inner_dim, depth=4, input_expand=False, tanh=True, **kwargs):
        '''
        :param vec_dim: 输出的表示向量维度
        :param inner_dim: 内部隐藏层的输出维度
        :param depth: 卷积残差块的数量
        :param tanh: 输出的表示向量是否经过tanh
        '''
        super(DilateEncoder2, self).__init__(**kwargs)
        self.input_expand = input_expand
        self.vec_dim = vec_dim
        self.inner_dim = inner_dim
        self.depth = depth
        self.list = [DilateBlock2(inner_dim, 1, need_resample=not input_expand)]
        dilation = 4
        for i in range(1, depth - 1):
            self.list.append(DilateBlock2(inner_dim, dilation))
            dilation *= 4
        self.list.append(DilateBlock2(vec_dim, dilation, need_resample=(vec_dim != inner_dim)))
        self.maxpool = kr.layers.GlobalMaxPool1D()
        activation = 'tanh' if tanh else None
        self.fcn = kr.layers.Dense(vec_dim, activation=activation, kernel_regularizer='l2')

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, 2) if self.input_expand else inputs
        for item in self.list:
            x = item(x, training=training)
        x = self.maxpool(x)
        return self.fcn(x)

    def get_config(self):
        return {'vec_dim': self.vec_dim, 'inner_dim': self.inner_dim,
                'depth': self.depth}


class TimeseriesModelCos(kr.Model):
    '''
    用于训练encoder的模型
    序列对输入该模型后，调用encoder模型计算向量，而后计算两个向量的"余弦相似性"，
        与label比对和梯度优化
    '''
    def __init__(self, encoder: kr.Model, **kwargs):
        super(TimeseriesModelCos, self).__init__(**kwargs)
        self.encoder = encoder
        self.dot = kr.layers.Dot(axes=1, normalize=True)

    def call(self, inputs, training=None, mask=None):
        line1, line2 = inputs['ts1'], inputs['ts2']
        vec1 = self.encoder(line1, training=training)
        vec2 = self.encoder(line2, training=training)
        return self.dot([vec1, vec2])

    def get_config(self):
        return {'encoder': self.encoder.get_config()}


class TimeseriesModelDist(kr.Model):
    '''
    用于训练encoder的模型
    序列对输入该模型后，调用encoder模型计算向量，而后计算两个向量的"欧氏距离"，
        与label比对和梯度优化
    '''
    def __init__(self, encoder: kr.Model, **kwargs):
        super(TimeseriesModelDist, self).__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs, training=None, mask=None):
        line1, line2 = inputs['ts1'], inputs['ts2']
        vec1 = self.encoder(line1, training=training)
        vec2 = self.encoder(line2, training=training)
        return tf.linalg.norm(kr.layers.subtract([vec1, vec2]), axis=1)

    def get_config(self):
        return {'encoder': self.encoder.get_config()}


class TimeseriesModelContrast(kr.Model):
    '''
    使用对比学习方法训练encoder的模型，损失函数是triplet loss。
    将原始数据输入该模型，将为每条时间序列取正样本、负样本，
        而后前向传播计算表示向量，后向梯度传播
    '''
    def __init__(self, encoder: kr.Model, train_data, min_len_ratio=0.5, neg_num=5, neg_penalty=1, **kwargs):
        '''
        :param min_len_ratio: 取正样本、负样本的最短长度比例（<=1），即最短长度不短于length* min_len_ratio
        :param neg_num: 负样本数量
        :param neg_penalty: loss中负样本的惩罚系数
        '''
        super(TimeseriesModelContrast, self).__init__(**kwargs)
        self.encoder = encoder
        self.dense = kr.layers.Dense(encoder.vec_dim, kernel_regularizer='l2')
        self.lrelu = kr.layers.LeakyReLU(0.01)
        self.train_data = train_data
        self.min_len_ratio = min_len_ratio
        self.neg_num = neg_num
        self.neg_penalty = neg_penalty
        self.dot = kr.layers.Dot(axes=1, normalize=True)

    def transform_emb(self, input):
        """
        需要在embedding的结果再过一层Dense和激活函数，最后再计算相似性
        这个原因是SimCLR之类的对比学习的套路：encoder提取的是全面的特征、
            需要将特征经过变换后，embedding的一小部分才有助于捕捉相似性
            然而下游任务可能需要保留全面的特征
        """
        x = self.dense(input)
        return self.lrelu(x)

    def call(self, inputs, training=None, mask=None):
        anchor_line, positive_line, neg_line = self.random_sample(inputs)

        # 计算表示向量
        anchor_emb = self.transform_emb(self.encoder(anchor_line, training=training))
        positive_emb = self.transform_emb(self.encoder(positive_line, training=training))
        # negative_embs = tf.map_fn(lambda buffer: self.transform_emb(self.encoder(buffer, training=training)),
        #                           neg_line, fn_output_signature=tf.float32)
        nlshape = tf.shape(neg_line)
        neg_line = tf.reshape(neg_line, (nlshape[0] * nlshape[1], nlshape[2], nlshape[3]))
        negative_embs = self.transform_emb(self.encoder(neg_line, training=training))
        negative_embs = tf.reshape(negative_embs, (nlshape[0], nlshape[1], tf.shape(negative_embs)[-1]))

        # Compute the loss value.
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_embs)
        if training:
            self.add_loss(loss)
            self.add_metric(loss, 'triplet loss')
        else:
            self.add_metric(loss, 'validate triplet loss')
        return anchor_emb, positive_emb, negative_embs

    def train_step(self, inputs):
        # Compute gradients
        with tf.GradientTape() as tape:
            anchor, positive, negative = self(inputs, training=True)
            loss = self.compiled_loss(None, None, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self(inputs, training=False)
        self.compiled_metrics.update_state(None, None)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def random_sample(self, inputs):
        '''从输入中随机生成正样本、负样本'''
        buffer_size = tf.shape(inputs)[0]
        # 负样本的位置
        samples = tf.random.uniform((buffer_size, self.neg_num), 0, self.train_data.shape[0],
                                    dtype=tf.int64)

        # 随机选择正、负样本的长度，在[输入长度 * min_len_ratio, 输入长度)间
        min_length = tf.math.ceil(self.train_data.shape[1] * self.min_len_ratio)
        length_pos_neg = tf.random.uniform([], tf.cast(min_length, tf.int64),
                                           self.train_data.shape[1], dtype=tf.int64)
        # anchor样本的长度
        anchor_length = tf.random.uniform([], length_pos_neg, self.train_data.shape[1] + 1, dtype=tf.int64)
        # anchor开始位置下标
        beginning_batches = tf.random.uniform([buffer_size], 0, self.train_data.shape[1] - anchor_length + 1,
                                              dtype=tf.int64)
        anchor_idx = tf.range(anchor_length) + tf.expand_dims(beginning_batches, axis=1)
        anchor_line = tf.gather(inputs, anchor_idx, batch_dims=1, axis=1)
        # 正样本开始位置下标
        beginning_positive = tf.random.uniform([buffer_size], 0, anchor_length - length_pos_neg + 1,
                                               dtype=tf.int64) + beginning_batches
        positive_idx = tf.range(length_pos_neg) + tf.expand_dims(beginning_positive, axis=1)
        positive_line = tf.gather(inputs, positive_idx, batch_dims=1, axis=1)
        # 负样本随机选择开始位置下标
        beginning_neg = tf.random.uniform((buffer_size, self.neg_num), 0,
                                          self.train_data.shape[1] - length_pos_neg + 1, dtype=tf.int64)
        neg_idx1 = tf.range(length_pos_neg) + tf.expand_dims(beginning_neg, axis=2)
        neg_idx = tf.stack([tf.broadcast_to(tf.expand_dims(samples, 2),
                                            (buffer_size, self.neg_num, tf.dtypes.cast(length_pos_neg, tf.int32))),
                            neg_idx1], axis=3)
        neg_line = tf.gather_nd(self.train_data, neg_idx)
        return anchor_line, positive_line, neg_line

    def triplet_loss(self, anchor, positive, negative):
        '''
        三元组计算loss函数。公式为：
        L = -log_sigmoid(anchor · positive) - ratio * sum_i(log_sigmoid(-anchor · negative_i))
        输入的是表示向量。输入anchor.shape = positive.shape = [batch, feature]
            negative.shape = [batch, negative_num, feature]
        '''
        loss = tf.reduce_mean(tf.nn.softplus(-self.dot([anchor, positive])))

        anchor = tf.expand_dims(anchor, axis=1)
        loss += self.neg_penalty * tf.reduce_mean(tf.nn.softplus(
            tf.matmul(anchor, negative, transpose_b=True)))

        # for i in range(self.neg_num):
        #     loss += self.neg_penalty * tf.reduce_mean(tf.nn.softplus(
        #         self.dot([anchor, negative[:, i, :]])))
        return loss


def create_data_sim(o_timeline_f, o_similar_f, d_timeline_f, d_similar_f):
    '''
    构造一个时间序列数据集，数据集大小为taz_num * (taz_num - 1) * 2，
        即两两不同时间序列的相似性(O和D的时间序列合在一起训练，也可只输入O或者D)
    :returns TF对象：(训练集, 标签)， 其中训练集 = [时间序列集1, 集合2]
    '''
    train1s = []
    train2s = []
    labels = []
    # O、D的曲线合在一起训练
    for timeline_f, similar_f in ((o_timeline_f, o_similar_f),
                                  (d_timeline_f, d_similar_f)):
        if timeline_f is None or similar_f is None:
            continue
        tline = np.load(timeline_f)
        taz_num = tline.shape[0]
        sim = np.load(similar_f)
        # 将时间序列转成两两相对的状态，并与相似矩阵对应
        train1 = tline.reshape((1, taz_num, tline.shape[1])).repeat(taz_num, axis=0)
        train1 = train1.reshape((taz_num * taz_num, tline.shape[1]))
        train2 = tline.reshape((taz_num, 1, tline.shape[1])).repeat(taz_num, axis=1)
        train2 = train2.reshape((taz_num * taz_num, tline.shape[1]))
        label = sim.reshape(taz_num * taz_num)
        # 去除自相似的数据（train1和train2对应行是同一时间序列）
        valid_index = [i for i in range(taz_num * taz_num) if i // taz_num != i % taz_num]
        train1s.append(train1[valid_index, :])
        train2s.append(train2[valid_index, :])
        labels.append(label[valid_index])
    if len(train1s) > 1:
        train1s = np.concatenate(train1s, axis=0)
        train2s = np.concatenate(train2s, axis=0)
        labels = np.concatenate(labels, axis=0)
    else:
        train1s = train1s[0]
        train2s = train2s[0]
        labels = labels[0]
    print(train1s.shape)
    # return {'ts1': train1s, 'ts2': train2s}, labels
    return tf.data.Dataset.from_tensor_slices(({'ts1': train1s, 'ts2': train2s}, labels))


def create_data(o_timeline_f, d_timeline_f):
    '''
    简单的时间序列数据集，仅仅是返回时间序列本身
    :param o_timeline_f:
    :param d_timeline_f:
    :return: shape = (sample_num, timeline_length, 2)
    '''
    data = []
    for timeline_f in (o_timeline_f, d_timeline_f):
        if timeline_f is None:
            continue
        timeline = np.load(timeline_f)
        data.append(timeline)
    if len(data) > 1:
        train = np.stack(data, axis=2)
    else:
        train = data[0]
    print(train.shape)
    return tf.constant(train)


def train_model(dataset, output, encoder, vec_dim, loss, epochs, batchsize=1024, **kwargs):
    '''
    训练模型，保存的函数
    :param dataset: 数据集，对于两两相似对比，格式为（{'1': 时间序列集合1, '2': 集合2}, 相似性标签）
                    对于triplet loss的数据集，只需要原始时间序列
    :param output: 模型输出文件夹
    :param encoder: str, 编码器类型：'cnn', 'linear', 'baseline'
    :param vec_dim: 输出TAZ向量的维度
    :param inner_dim: 隐藏层内部维度
    :param loss: 向量间距离的度量方式：'cos'、'dist'（欧氏距离）、'triplet'（三元组损失函数）
    :param epochs: 训练周期数
    :param batchsize: batch大小
    :param depth: 神经网络的模块层数
    :param min_len_ratio: 仅用于triplet loss，设置正负样本取的最短长度比例
    '''
    # 选择编码器
    if 'inner_dim' not in kwargs:
        kwargs['inner_dim'] = vec_dim
    if encoder.lower() == 'cnn':
        # 计算cos相似性时，输出向量的norm应该不要太大
        encoder_model = DilateEncoder(vec_dim, inner_dim=kwargs['inner_dim'], depth=kwargs['depth'],
                                      tanh=loss.lower() != 'dist', name='cnn')
    elif encoder.lower() == 'cnn2':
        # 计算cos相似性时，输出向量的norm应该不要太大
        # 这个Encoder的总层数应该更短
        encoder_model = DilateEncoder2(vec_dim, inner_dim=kwargs['inner_dim'], depth=kwargs['depth'],
                                       tanh=loss.lower() != 'dist', name='cnn2')
    elif encoder.lower() == 'rnn':
        encoder_model = GRUEncoder(vec_dim, depth=kwargs['depth'], tanh=loss.lower() != 'dist', name='rnn')
    elif encoder.lower() == 'linear':
        encoder_model = create_linearmodel(7 * 24, vec_dim)
    elif encoder.lower() == 'baseline':
        encoder_model = create_basemodel(7 * 24, vec_dim)
    else:
        raise ValueError('not implement except `cnn`, `cnn2`, `rnn`, `linear` and `baseline`')
    # 选择损失函数
    if loss.lower() == 'cos':
        model = TimeseriesModelCos(encoder_model)
    elif loss.lower() == 'dist':
        model = TimeseriesModelDist(encoder_model)
    elif loss.lower() == 'triplet':
        if 'min_len_ratio' not in kwargs:
            kwargs['min_len_ratio'] = 0.5
        model = TimeseriesModelContrast(encoder_model, dataset, min_len_ratio=kwargs['min_len_ratio'])
    else:
        raise ValueError('not implement except `cos` and `dist`')

    # 外部模型参数
    optimizer = kr.optimizers.Adam(learning_rate=1e-3)
    callbacks = []
    if isinstance(model, TimeseriesModelContrast):
        def schedule_lr(ep, lr):
            """
            调整学习率的函数，输入当前周期数和`当前`的学习率，输出调整后的学习率
            """
            if ep < epochs / 2:
                return lr
            return lr * 0.8 if ep % 10 == 0 else lr

        model.compile(optimizer=optimizer, loss=None)
        # callbacks.append(kr.callbacks.Le
        # arningRateScheduler(schedule_lr))
    else:
        model.compile(optimizer=optimizer, loss=kr.losses.MeanSquaredError())

    if not isinstance(dataset, tf.data.Dataset):
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.shuffle(buffer_size=600*1200).batch(batchsize)
    history = model.fit(dataset, epochs=epochs, workers=4, callbacks=callbacks)

    # 模型保存输出
    if not os.path.exists(output):
        os.mkdir(output)
    # model.encoder.save(output)
    # model.encoder.save_weights(output)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv((output[:-1] if output[-1] == '/' else output) + '.csv')
    return model


def predict_model(model, datafile, output):
    '''
    模型后处理：从给定时间序列计算向量表示，以及向量的特征，最后保存向量
    '''
    if isinstance(datafile, str):
        data = np.load(datafile)
    else:
        data = np.stack([np.load(f) for f in datafile], axis=-1)
    result = model.encoder.predict(data)
    # 计算向量的模长和分布
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    print({'max': np.max(norm), 'min': np.min(norm), 'mean': np.mean(norm)})
    normdata = result / norm
    # 计算相似矩阵及其分布
    sim = np.dot(normdata, normdata.T)
    print({'max': np.max(sim), 'min': np.min(sim), 'mean': np.mean(sim)})
    np.save(output, result)


def baseline(dataset, vec_dim, loss):
    '''
    损失函数的基准：用PCA降维，降维后向量计算欧氏距离，而后线性回归的方式，拟合DTW或相似性，得到基准loss
    证明深度网络模型是更优的
    dist的基准loss：vec_dim=7*24: 0.6554497432621165; vec_dim=64: 0.7032972554132926
    :param dataset: 时间序列数据集，从`create_7x24data`函数中生成
    :param vec_dim: 降维后的维度
    :param loss: 向量间距离的度量方式：'cos'或'dist'
    '''
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # PCA降维
    pca = PCA(vec_dim)
    datats1 = dataset[0]['ts1'].squeeze()
    datats2 = dataset[0]['ts2'].squeeze()
    pca.fit(datats1)
    ts1 = pca.transform(datats1)
    ts2 = pca.transform(datats2)
    # 选择loss计算方式
    if loss == 'cos':
        ts1 = ts1 / np.linalg.norm(ts1, axis=1, keepdims=True)
        ts2 = ts2 / np.linalg.norm(ts2, axis=1, keepdims=True)
        diff = np.sum(ts1 * ts2, axis=1)
    elif loss == 'dist':
        diff = np.linalg.norm(ts2 - ts1, axis=1)
    else:
        raise ValueError('not implement except `cos` and `dist`')
    diff = diff.reshape((diff.shape[0], 1))
    # 线性拟合并计算loss
    lg = LinearRegression()
    lg.fit(diff, dataset[1])
    loss = np.mean(np.square(lg.predict(diff) - dataset[1]))
    print('loss:', loss, 'coef:', lg.coef_, 'intercept:', lg.intercept_)
    return loss


def create_linearmodel(timeseries_len, vec_dim):
    '''
    另一基准模型：梯度下降的线性变换模型（用全连接层实现）
    这个模型在vec_dim=64时: 最后batch的loss=0.2783，全局loss=0.2494
    :param timeseries_len: 时间序列长度
    :param vec_dim: 输出向量维数
    '''
    return kr.Sequential([kr.layers.Flatten(input_shape=(timeseries_len, 1)),
                          kr.layers.Dense(vec_dim)])


def create_basemodel(timeseries_len, vec_dim):
    '''
    另一基准模型：简单的全连接层+非线性输出
    这个模型在vec_dim=64时: 最后batch的loss=0.2406，全局loss=0.2106
    :param timeseries_len: 时间序列长度
    :param vec_dim: 输出向量维数
    '''
    return kr.Sequential([kr.layers.Flatten(input_shape=(timeseries_len, 1)),
                          kr.layers.Dense(vec_dim),
                          kr.layers.LeakyReLU(0.01)])


if __name__ == '__main__':
    vecdim = 32
    inner_dim = 32
    encoder = 'cnn'
    epoch = 100
    daynum = 7      # 可以是2或者7（表示时间序列是2*24或7*24）
    if encoder == 'cnn2':
        depth = 2 if daynum == 2 else 3     # 网络层数
    else:
        depth = 3 if daynum == 2 else 4  # 网络层数
    # depth = 3
    # timeline_norm = 'abs'   # 有效值：'abs'（所有序列同时除一个值）和'rel'（每条序列分别z-score）
    loss = 'triplet'
    minlen = 4
    # 训练1：用cos刻画向量相似性
    # dataset = create_data(file_root + 'timeline/o_timeline_7x24.npy',
    #                       file_root + 'timeline/o_timeline_sim_7x24_r2.npy',
    #                       file_root + 'timeline/d_timeline_7x24.npy',
    #                       file_root + 'timeline/d_timeline_sim_7x24_r2.npy')
    # model = train_model(dataset, file_root + 'cnnv2cos_d64_ep10_deep6', 'cnn',
    #                     vec_dim=64, inner_dim=48, loss='cos', epochs=10)

    # 训练2：用dist刻画向量相似性
    # simstr = 'dtw'
    # dataset = create_data_sim(f'{file_root}timeline/o_timeline_{daynum}x24.npy',
    #                           f'{file_root}timeline/o_timeline_{simstr}_{daynum}x24.npy'
    #                           f'{file_root}timeline/d_timeline_{daynum}x24.npy',
    #                           f'{file_root}timeline/d_timeline_{simstr}_{daynum}x24.npy')
    # model = train_model(dataset, f'{file_root}{encoder}_{daynum}x24_{simstr}_d{vecdim}_ep{epoch}_deep5', encoder,
    #                     vec_dim=vecdim, inner_dim=48, loss='dist', epochs=epoch)

    # 训练3：用triplet loss提取正负样本，而后训练表示向量
    for timeline_norm in ['abs', 'rel', 'log']:
        dataset = create_data(f'{file_root}timeline/o_timeline_{daynum}x24{timeline_norm}.npy',
                              f'{file_root}timeline/d_timeline_{daynum}x24{timeline_norm}.npy')
        model = train_model(dataset, f'{file_root}{encoder}_{daynum}x24{timeline_norm}_{loss}_d{vecdim}_ep{epoch}_deep{depth}_ml{minlen}',
                            encoder, vec_dim=vecdim, depth=depth, inner_dim=inner_dim,
                            loss=loss, epochs=epoch, batchsize=584 // 2, min_len_ratio=(minlen/(24*daynum)-1e-6))

        # 推理数据
        model.encoder.summary()
        # predict_model(model, f'{file_root}timeline/o_timeline_{daynum}x24{timeline_norm}.npy',
        #               f'{file_root}ts_vec/o{daynum}x24{timeline_norm}_{loss}_d{vecdim}_ep{epoch}.npy')
        # predict_model(model, f'{file_root}timeline/d_timeline_{daynum}x24{timeline_norm}.npy',
        #               f'{file_root}ts_vec/d{daynum}x24{timeline_norm}_{loss}_d{vecdim}_ep{epoch}.npy')
        # 合并输入数据的推理
        predict_model(model, [f'{file_root}timeline/o_timeline_{daynum}x24{timeline_norm}.npy',
                              f'{file_root}timeline/d_timeline_{daynum}x24{timeline_norm}.npy'],
                      f'{file_root}ts_vec/od{daynum}x24{timeline_norm}_{encoder}_{loss}_d{vecdim}_ep{epoch}_ml{minlen}.npy')

    # 基准测试：
    # baseline(dataset, 64, 'dist')
    # model = train_model(dataset, file_root + 'linear_d64_ep10', 'linear',
    #                     vec_dim=vecdim, inner_dim=48, loss='dist', epochs=10)
