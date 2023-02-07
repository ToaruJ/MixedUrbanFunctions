import pandas as pd
import gensim.models.callbacks as gscb
import gensim.models.doc2vec as d2v
import numpy as np
import time
import os


doc_name = 'E:/gis_data/微博/Sina2016Weibo（处理好的）/5ring_weibo_result2.csv'
model_root = 'E:/gis_data/微博/Sina2016Weibo（处理好的）/'


class EpochLoggerDV(gscb.CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self, val=None, recall=None):
        self.epoch = 0
        self.start_time = None
        self.val = val          # 验证集
        self.recall = recall    # 召回数（相似前recall个算命中）

    def on_epoch_begin(self, model):
        self.dv = model.dv.vectors.copy()
        self.start_time = time.time()

    def on_epoch_end(self, model):
        print("Epoch %d end. time use: %.2fs, " % (self.epoch, time.time() - self.start_time), end='')
        if self.val is not None:
            # 验证集验证效果代码：用训练集本身来验证是否可行？还是必须要验证集？
            self.val = pd.Series()
            result = self.val.apply(self._validate, args=(model,))
            result = result.value_counts(normalize=True)
            recall = result[True] if True in result.index else 0
            print('recall = %.6f' % recall)
        else:
            change = np.max(np.linalg.norm(model.dv.vectors - self.dv, axis=1))
            loss = model.get_latest_training_loss()
            print('change = %.6f, loss = %.6f' % (change, loss))
        self.epoch += 1
        # if self.epoch == 1 or self.epoch % 2 == 0:
        #     model.save(self.root + 'ep%02d.d2v' % self.epoch)

    def _validate(self, line, model):
        vec = model.infer_vector(line.words)
        sim = model.dv.similar_by_vector(vec, self.recall)
        return line.tags[0] in list(zip(*sim))[0]


def doc_justTAZ(filename):
    '''
    数据集1：其中的文档只按照TAZ分割，不区分发送时间
    文档ID = TAZ的ID
    '''
    frame = pd.read_csv(filename, usecols=['TAZ_ID', 'weibo_fenci'])
    frame.dropna(inplace=True)
    frame['doc'] = frame.apply(lambda l: d2v.TaggedDocument(
        l['weibo_fenci'].split(' '), [l['TAZ_ID'], ]), axis=1)
    del frame['weibo_fenci']
    print('weibo num:', frame.shape[0])
    return frame


def doc_TAZ2x12(filename):
    '''
    数据集2：其中文档按照TAZ，周间or周末，小时（2h一组）分割。共len(TAZ)*2*12个文档
    文档ID = TAZ的ID * 24 + {周间0, 周末12} + 小时 // 2
    '''
    frame = pd.read_csv(filename, usecols=['TAZ_ID', 'weekday', 'hour', 'weibo_fenci'])
    frame.dropna(inplace=True)
    frame['doc'] = frame.apply(lambda l: d2v.TaggedDocument(
        l['weibo_fenci'].split(' '),
        [l['TAZ_ID'] * 24 + (0 if l['weekday'] < 5 else 12) + l['hour'] // 2, ]), axis=1)
    del frame['weibo_fenci']
    return frame


def doc_TAZ7x12(filename):
    '''
    数据集3：其中文档按照TAZ，周几，小时（2h一组）分割。共len(TAZ)*7*12个文档
    文档ID = TAZ的ID * 84 + 周几(Mon.=0, Sun.=6) * 12 + 小时 // 2
    '''
    frame = pd.read_csv(filename, usecols=['TAZ_ID', 'weekday', 'hour', 'weibo_fenci'])
    frame.dropna(inplace=True)
    frame['doc'] = frame.apply(lambda l: d2v.TaggedDocument(
        l['weibo_fenci'].split(' '),
        [l['TAZ_ID'] * 84 + l['weekday'] * 12 + l['hour'] // 2, ]), axis=1)
    del frame['weibo_fenci']
    return frame


def train_D2V(doc, title, vector_size=64, epoch=40):
    '''
    训练Doc2Vec模型：默认迭代20次，特征维数64
    已训练有48维，64维
    '''
    logger = EpochLoggerDV()
    model = d2v.Doc2Vec(doc['doc'], vector_size=vector_size, negative=5, dm=0,
                        window=5, sample=0, epochs=epoch, workers=8,
                        callbacks=[logger], compute_loss=True)
    # model.build_vocab(doc['doc'])
    # print('prepare for train')
    # model.train(doc['doc'], total_examples=model.corpus_count, epochs=epoch,
    #             callbacks=[logger])
    model.save(model_root + '%s_size%d_ep%d.d2v' % (title, vector_size, epoch))
    vectors = np.zeros(model.dv.vectors.shape)
    for i in range(model.dv.vectors.shape[0]):
        vectors[i, :] = model.dv.get_vector(i)
    np.save(model_root + ('doc2vec_vectors/%s_size%d_ep%d.npy' %
                          (title, vector_size, epoch)), vectors)
    return model


def trainval_D2V(doc: pd.DataFrame, title, frac=0.8, recall=5, vector_size=64, epoch=40):
    '''
    训练Doc2Vec模型：默认迭代20次，特征维数64
    增加对训练结果的评价：使用一定比例（默认0.2，frac是训练集比例）验证集
    recall是验证的召回率计算：最相似的文档数。
    '''
    doc['sample'] = np.random.random(doc.shape[0])
    logger = EpochLoggerDV(doc[doc['sample'] >= frac]['doc'], recall)
    model = d2v.Doc2Vec(vector_size=vector_size, negative=5,
                        window=5, sample=0, epochs=epoch, workers=8,
                        callbacks=[logger])
    model.build_vocab(doc['doc'])
    model.train(doc[doc['sample'] < frac]['doc'], total_examples=model.corpus_count,
                epochs=model.epochs, callbacks=[logger], compute_loss=True)
    model.save(model_root + '%s_size%d_ep%d.d2v' % (title, vector_size, epoch))
    vectors = np.zeros(model.dv.vectors.shape)
    for i in range(model.dv.vectors.shape[0]):
        vectors[i, :] = model.dv.get_vector(i)
    np.save(model_root + ('doc2vec_vectors/%s_size%d_ep%d.npy' %
                          (title, vector_size, epoch)), vectors)
    return model


def similarity_test():
    '''
    TAZ ID = 538是北大
    justTAZ模型对比最相似的TAZ，TAZ2x12对比相似的(TAZ+周间周末+hour)，TAZ7x12对比相似时段
    '''
    model0 = d2v.Doc2Vec.load(model_root + 'justTAZ_size64_ep20.d2v')
    print(model0.dv.similar_by_key(538))

    # doc = doc_justTAZ(doc_name)
    # frame = pd.read_csv(doc_name, usecols=['TAZ_ID', 'weibo_fenci'])
    # frame.dropna(inplace=True)
    # val = frame.sample(n=10000)
    # val['doc'] = val.apply(lambda l: d2v.TaggedDocument(
    #     l['weibo_fenci'].split(' '), [l['TAZ_ID'], ]), axis=1)
    # def valid(line, model, n_recall):
    #     vec = model.infer_vector(line.words)
    #     sim = model.dv.similar_by_vector(vec, n_recall)
    #     # print({'tag:': line.tags, 'sim:': sim})
    #     return line.tags[0] in list(zip(*sim))[0]
    # recall = val['doc'].apply(valid, args=(model0, 10))
    # cnt = recall.value_counts(normalize=True)
    # recall = cnt[True] if True in cnt.index else 0
    # print('recall = %.6f' % recall)

    # model1 = d2v.Doc2Vec.load(model_root + 'TAZ2x12_size64_ep20.d2v')
    # # 比较周末20~22h
    # result1 = model1.dv.similar_by_key(538 * 24 + 12 + 10)
    # print([{'taz:': i // 24, 'isweekend:': i % 24 >= 12, 'hour:': i % 12}
    #        for i, sim in result1])
    #
    # model2 = d2v.Doc2Vec.load(model_root + 'TAZ7x12_size64_ep20.d2v')
    # # 比较周六20~22h
    # result2 = model2.dv.similar_by_key(538 * 84 + 12 * 5 + 10)
    # weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    # print([{'taz:': i // 84, 'weekday:': weekday[(i % 84) // 12], 'hour:': i % 12}
    #        for i, sim in result2])

    # print(model0.wv.similar_by_key('股票'))


def model2vector(fileroot, output):
    '''
    将模型中的向量输出
    '''
    listdir = os.listdir(fileroot)
    for f in listdir:
        f_abs = os.path.join(fileroot, f)
        if os.path.isfile(f_abs):
            base, ext = os.path.splitext(f)
            if ext == '.d2v':
                model = d2v.Doc2Vec.load(f_abs)
                vec = model.dv.vectors
                np.save(os.path.join(output, base + '.npy'), vec)


if __name__ == '__main__':
    doc = doc_justTAZ(doc_name)
    model = train_D2V(doc, 'justTAZdbow', vector_size=32, epoch=40)
    # doc = doc_TAZ2x12(doc_name)
    # model = train_D2V(doc, 'TAZ2x12', vector_size=80, epoch=20)
    # doc = doc_TAZ7x12(doc_name)
    # model = train_D2V(doc, 'TAZ7x12', vector_size=80, epoch=20)
    # similarity_test()
    # model2vector(model_root, model_root + 'doc2vec_vectors/')
