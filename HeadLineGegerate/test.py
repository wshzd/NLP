#!/usr/bin/python
# -*- coding:utf-8 -*-

import random
import tqdm

data = [[[23, 883, 185,32,777,888], [18, 66, 23]], [[103, 32, 154], [69, 31, 536, 31]], [[8, 569, 9290, 154], [6, 6790, 4658]]]
data_len = len(data)
batch_size = 2
encoder_inputs = []
decoder_targets = []
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

        #将target进行PAD，并添加END符号
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getBatches(data, batch_size):
    #每个epoch之前都要进行样本的shuffle
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
batches = getBatches(data, batch_size)
print(batches[0].encoder_inputs)
print(batches[1].encoder_inputs)
