import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os

tf.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')
tf.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.flags.DEFINE_integer('min_freq', 0, 'Minimum # of word frequency')
tf.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.flags.DEFINE_string('tmp_dir', 'chatbot/demo', 'Path to save model checkpoints')
tf.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.flags.DEFINE_string('data_path', 'test1.txt', 'data_dir')
FLAGS = tf.flags.FLAGS

word2id, id2word, trainingSamples = loadDataset(FLAGS.data_path,FLAGS.min_freq)
with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())
    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        print("trainingSamples----------------", trainingSamples)
        batches = getBatches(trainingSamples, FLAGS.batch_size)
        print('batches----------',batches)
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary = model.train(sess, nextBatch)
            current_step += 1
            #if current_step % FLAGS.steps_per_checkpoint == 0:
            if current_step % 2 == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)
                
