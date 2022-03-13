#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import math
import sys
import tempfile
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("classification_data_file", "./data/rt-polaritydata/mergeAllclass.txt", "Data source for the classification raw data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
#tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_string("filter_sizes", "5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 4, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 2, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Distribute Parameters
tf.flags.DEFINE_boolean("sync_replicas", False,"Use the sync_replicas (synchronized replicas) mode, wherein the parameter updates from workers are aggregated before applied to avoid stale gradients")
tf.flags.DEFINE_boolean("existing_servers", False, "Whether servers already exists. If True, will use the worker hosts via their GRPC URLs (one client process per worker host). Otherwise, will create an in-process TensorFlow server.")
tf.flags.DEFINE_string("ps_hosts", "localhost:2222", "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "Comma-separated list of hostname:port pairs")
tf.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.flags.DEFINE_integer("task_index", None, "Worker task index, should be >= 0. task_index=0 is the master worker task the performs the variable initialization")
tf.flags.DEFINE_integer("num_gpus", 1,"Total number of gpus for each machine.If you don't use GPU, please set it to '0'")
tf.flags.DEFINE_integer("replicas_to_aggregate", None, "Number of replicas to aggregate before parameter update is applied (For sync_replicas mode only; default: num_workers)")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Preparation Distribution config
# ===============================================================

# Check whether job_name and task_index is exist
if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")
print("job name = %s" % FLAGS.job_name)
print("task index = %d" % FLAGS.task_index)

# 创建当前task结点的Server
# Construct the cluster and start the server
ps_spec = FLAGS.ps_hosts.split(",")
worker_spec = FLAGS.worker_hosts.split(",")
# Get the number of workers.
num_workers = len(worker_spec)
# 创建cluster and server
cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
# 如果当前节点是ps，则调用server.join()无休止等待；
if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
# 判断是否为主节点
is_chief = (FLAGS.task_index == 0)
# worker节点计算资源配置
if FLAGS.num_gpus > 0:
    # if FLAGS.num_gpus < num_workers:
    #  raise ValueError("number of gpus is less than number of workers")
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

# The device setter will automatically place Variables ops on separate
# parameter servers (ps). The non-Variable ops will be placed on the workers.
# The ps use CPU and workers use corresponding GPU
with tf.device(
    tf.train.replica_device_setter(
        worker_device=worker_device,
        ps_device="/job:ps/cpu:0",
        cluster=cluster)):

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    # x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    x_text, y = data_helpers.process_chinese(FLAGS.classification_data_file)
    print(y[:5])
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # vocab_processor = learn.preprocessing.VocabularyProcessor(100)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = np.array(y)[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print('vocabulary大小',len(vocab_processor.vocabulary_))

    # Training
    # ==================================================
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)

    cnn = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    timestamp = str(int(time.time()))
    # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_merge", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    # Write vocabulary
    # vocab_processor.save(os.path.join(out_dir, "vocab"))


    # 本地参数初始化
    init_op = tf.global_variables_initializer()
    # 临时训练目录
    train_dir = tempfile.mkdtemp()
    # 同步模式需要对优化器进行扩展
    if FLAGS.sync_replicas:
        # n batch后更新模型参数
        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate
        # 创建新的优化器
        opt = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=replicas_to_aggregate,
                                             total_num_replicas=num_workers, name="textcnn_sync_replicas")
        # 初始化
        local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op
        ready_for_local_init_op = opt.ready_for_local_init_op
        # 队列执行器
        chief_queue_runner = opt.get_chief_queue_runner()
        # 全局参数初始化器
        sync_init_op = opt.get_init_tokens_op()
        # 创建tf.train.Supervisor来管理模型的训练过程
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            local_init_op=local_init_op,
            ready_for_local_init_op=ready_for_local_init_op,
            recovery_wait_secs=1,
            global_step=global_step)
    else:
        sv = tf.train.Supervisor(
            is_chief=is_chief,
            logdir=train_dir,
            init_op=init_op,
            recovery_wait_secs=1,
            global_step=global_step)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                 device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # 准备运行
    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)
    # 创建session
    if FLAGS.existing_servers:
        server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
        print("Using existing server at: %s" % server_grpc_url)
        sess = sv.prepare_or_wait_for_session(server_grpc_url,config=sess_config)
    else:
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
        # Chief worker will start the chief queue runner and call the init op.
        sess.run(sync_init_op)
        sv.start_queue_runners(sess, [chief_queue_runner])

    # Perform training 开始训练
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    # Initialize all variables

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, loss, accuracy = sess.run(
            [train_op, global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch):
        """
        Evaluates model on a dev set
        """
        import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 为空的时候，代表不使用GPU
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, loss, accuracy = sess.run(
            [global_step, cnn.loss, cnn.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # if writer:
        #     writer.add_summary(summaries, step)

    # Generate batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs, False)
    # Training loop. For each batch...
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            evaluation_batchs = data_helpers.batch_iter(
                list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
            print("\nEvaluation:")
            for index, evaluation_batch in enumerate(evaluation_batchs):
                x_evaluation, y_evaluation = zip(*evaluation_batch)
                # dev_step(x_evaluation, y_evaluation, writer=dev_summary_writer)
                dev_step(x_evaluation, y_evaluation)
                print("current batch {}".format(index + 1))
                print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))





