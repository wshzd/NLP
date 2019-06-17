import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import os
import numpy as np

tf.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.flags.DEFINE_integer('min_freq', 0, 'Minimum # of word frequency')
tf.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.flags.DEFINE_string('data_path', 'test1.txt', 'data_dir')
FLAGS = tf.flags.FLAGS

word2id, id2word, trainingSamples = loadDataset(FLAGS.data_path,FLAGS.min_freq)

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    # if os.path.exists(FLAGS.model_dir):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

    total_sentence_list = []
    with open(FLAGS.data_path, 'r', encoding='utf8') as file_object:
        for line in file_object.readlines():
            one_sentence_list = []
            if (len(line)) > 12:
                loc = locals()
                exec('b=' + line)
                for content in loc['b'][0]:
                    one_sentence_list.append(content['feature'])
                total_sentence_list.append(one_sentence_list)

    for sentence in total_sentence_list:
        batch = sentence2enco(sentence, word2id)
        predicted_ids = model.infer(sess, batch)
        predict_ids_to_seq(predicted_ids, id2word, 3)
        
