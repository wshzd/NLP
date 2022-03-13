from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pickle
import random

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3
class Batch:
    # batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []
        self.word2id_dict = {}
        self.id2word_dict = {}
        self.START_ID = 4
        self.special_word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unknown>':3}

batch = Batch()
# loadDataset函数我只写了一部分，读者可以自己补充完成编写，主要是输入文件数据，然后返回字典：词2id、id2词和列表：每一行数据中词对应的id
def loadDataset(filename,min_freq):
    """
    加载样本文件列表，全部切词后统计词频，按词频由高到低排序后顺次编号
    并存到self.word2id_dict和self.id2word_dict中
    """
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    words_count = {}
    # 统计文件中词频大于min_freq的词，并且生成word2id和id2word
    
    word2id = dict(batch.word2id_dict,**batch.special_word2id)
    id2word = dict([[v[1], v[0]] for v in word2id.items()])

    # 转换句子为ID
    trainingSamples = []
    with open(filename, 'r', encoding='utf8') as file_object:
        
    return word2id,id2word,trainingSamples

def createBatch(samples):
    '''
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        # 将source进行反序并PAD值本batch的最大长度
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        # 将target进行PAD，并添加END符号
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getBatches(data, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''
    # 每个epoch之前都要进行样本的shuffle
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''


    if sentence == '':
        return None
    # 分词

    # 将每个单词转化为id
    wordIds = []
    for token in sentence:
        wordIds.append(word2id.get(token, unknownToken))
    # 调用createBatch构造batch
    batch = createBatch([[wordIds, []]])
    return batch
