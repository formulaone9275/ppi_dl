from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from datetime import datetime
from constant import *
#from tfrecord_reader_attn import iter_sent_dataset
from word_embedding import VOCAB_SIZE, EMBEDDING
from flags import *
import os, pickle
from tqdm import tqdm
from glob import glob
import random

def post_process_seq(seq):
    # First 5 element in a 155-dim token:
    # word_index, bin_token, bin_other, bin_p1, bin_p2.
    # See data_util_new.py for details.
    seq_len = int(seq.shape[0])

    if tf.flags.FLAGS.mask_p1p2:
        p1_col = tf.equal(seq[:, 1:2], 3)
        p2_col = tf.equal(seq[:, 1:2], 4)
        p1p2_col = tf.equal(seq[:, 1:2], 5)
        p1p2 = tf.logical_or(p1_col, p2_col)
        p1p2 = tf.logical_or(p1p2, p1p2_col)
        not_p1p2 = tf.cast(tf.logical_not(p1p2), tf.float32)
        word, feature = tf.split(seq, [1, 5], axis=1)
        new_word = tf.multiply(word, not_p1p2)
        # mask it as _ARG_ENTITY_
        new_word = new_word + tf.cast(p1p2, tf.float32) * 3
        seq = tf.concat([new_word, feature], axis=1)

    if tf.flags.FLAGS.mask_other:
        other_col = tf.equal(seq[:, 1:2], 2)
        not_other = tf.cast(tf.logical_not(other_col), tf.float32)
        word, feature = tf.split(seq, [1, 5], axis=1)
        new_word = tf.multiply(word, not_other)
        # mask it as _ENTITY_ words.
        new_word = new_word + tf.cast(other_col, tf.float32) * 2
        seq = tf.concat([new_word, feature], axis=1)

    if tf.flags.FLAGS.other_encoding == 'normal':
        other_col = tf.cast(tf.equal(seq[:, 1:2], 2), tf.float32)
        not_other_col = tf.cast(tf.not_equal(seq[:, 1:2], 2), tf.float32)

        normal_col = tf.constant([[1]]*seq_len, dtype=tf.float32)
        mask_col = tf.multiply(other_col, normal_col)
        word, entity_col, rest = tf.split(seq, [1, 1, 4], axis=1)
        mask_entity_col = tf.multiply(entity_col, not_other_col)
        final_entity = mask_entity_col + mask_col
        seq = tf.concat([word, final_entity, rest], axis=1)

    if tf.flags.FLAGS.vocab_size > 0:
        word_col = tf.less_equal(seq[:, 0:1], tf.flags.FLAGS.vocab_size)
        rare_word_col = tf.cast(tf.logical_not(word_col), tf.float32)
        word_col = tf.cast(word_col, tf.float32)
        word, rest = tf.split(seq, [1, 5], axis=1)
        new_word = tf.multiply(word, word_col)
        new_word = new_word + rare_word_col
        seq = tf.concat([new_word, rest], axis=1)

    return seq


def _sent_parse_func(example_proto):
    features = {
        'seq': tf.FixedLenFeature([340*6,], tf.float32),
        'seq_len': tf.FixedLenFeature([3,], tf.float32),
        'entity': tf.FixedLenFeature([12,], tf.float32),
        'label': tf.FixedLenFeature([2,], tf.float32),
    }
    parsed = tf.parse_single_example(example_proto, features)
    seq = tf.reshape(parsed['seq'], [340, 6])
    seq = post_process_seq(seq)

    entity = tf.reshape(parsed['entity'], [2, 6])
    entity = post_process_seq(entity)

    sent, head, dep = tf.split(seq, [160, 160, 20], axis=0)
    sent_len, head_len, dep_len = tf.split(parsed['seq_len'], [1, 1, 1])

    return (sent, sent_len, head, head_len, dep, dep_len, entity, parsed['label'])


# data.shuffle(buffer_size=130000) is too slow, if we use smaller buffer, we
# can't get perfect shuffle of the whole data set. So we split the tfrecord
# files into smaller ones and shuffle filenames at each epoch.
def iter_sent_dataset(sess, filename, epoch, batch_size, shuffle=True):
    ph_files = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(ph_files)
    dataset = dataset.map(_sent_parse_func, num_parallel_calls=8)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)

    files = glob(filename)
    for e in range(epoch):
        random.shuffle(files)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        sess.run(iterator.initializer, {ph_files: files})

        while True:
            try:
                batch = sess.run(next_element)
                sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, entity, labels = batch
                # To make it (?, 1).
                sent_len = np.ravel(sent_len)
                head_len = np.ravel(head_len)
                dep_len = np.ravel(dep_len)
                yield sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, entity, labels
            except tf.errors.OutOfRangeError:
                break

def split_tfrecord_file(filename, target, num_per_split):
    pre_filename = os.path.basename(filename).split('.')[0]
    curr_slice = 0
    slice_filename = '{}/{}_{}.tfrecords'.format(target, pre_filename, curr_slice)
    writer = tf.python_io.TFRecordWriter(slice_filename)

    for count, r in enumerate(tqdm(tf.python_io.tf_record_iterator(filename))):
        writer.write(r)
        if count > 0 and count % num_per_split == 0:
            writer.close()
            curr_slice += 1
            slice_filename = '{}/{}_{}.tfrecords'.format(target, pre_filename, curr_slice)
            writer = tf.python_io.TFRecordWriter(slice_filename)

    if count % num_per_split > 0:
        writer.close()


class CNNModel(object):
    def __init__(self):
        FLAGS = tf.flags.FLAGS
        self.emb_dim = FLAGS.emb_dim
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.num_kernel = FLAGS.num_kernel
        self.min_window = FLAGS.min_window
        self.max_window = FLAGS.max_window
        self.l2_reg = FLAGS.l2_reg
        self.lr = FLAGS.lr
        self.lr_decay_step = FLAGS.decay_step
        self.lr_decay_rate = FLAGS.decay_rate
        self.use_head = FLAGS.use_head
        self.use_dep = FLAGS.use_dep
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

    def conv2d(self, name, inputs, window_size):
        inputs = tf.expand_dims(inputs, -1)
        feature_dim = int(inputs.shape[-2])
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=self.num_kernel,
                                kernel_size=[window_size, feature_dim],
                                activation=tf.nn.relu,
                                kernel_regularizer=self.regularizer,
                                padding='valid',
                                name=name + '-conv',
                                trainable=self.trainable)

        batch_norm = tf.layers.batch_normalization(
            conv,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training,
            trainable=self.trainable,
            name=name+'-batch-norm')

        '''
        conv_len = int(conv.shape[1])
        pool = tf.layers.max_pooling2d(inputs=batch_norm,
                                       pool_size=[conv_len, 1],
                                       strides=1,
                                       padding='valid',
                                       name=name + '-pool')
        pool_size = self.num_kernel
        flat = tf.reshape(pool, [-1, pool_size], name=name+"-flat")
        '''
        return batch_norm

    def embed_feature(self, seq, name):
        length = int(seq.shape[1])
        token, enttype, pos, toe1, toe2, dep = tf.split(seq, [1]*6, 2, name=name + '-split')

        token = tf.cast(tf.reshape(token, [-1, length]), tf.int32)
        pos = tf.cast(tf.reshape(pos, [-1, length]), tf.int32)
        enttype = tf.cast(tf.reshape(enttype, [-1, length]), tf.int32)
        toe1 = tf.cast(tf.reshape(toe1, [-1, length]), tf.int32)
        toe2 = tf.cast(tf.reshape(toe2, [-1, length]), tf.int32)
        dep = tf.cast(tf.reshape(dep, [-1, length]), tf.int32)

        embed_token = tf.nn.embedding_lookup(self.embedding_weights, token)
        embed_pos = tf.nn.embedding_lookup(self.pos_weights, pos)
        embed_enttype = tf.nn.embedding_lookup(self.entity_type_weights, enttype)
        embed_toe1 = tf.nn.embedding_lookup(self.distance_weights, toe1)
        embed_toe2 = tf.nn.embedding_lookup(self.distance_weights, toe2)

        embed_dep = tf.nn.embedding_lookup(self.dep_weights, dep)

        final = tf.concat([embed_token, embed_pos,
                           embed_enttype, embed_toe1, embed_toe2,
                           #on_dep,
                           embed_dep],
                          axis=2, name=name+'-final')

        return final

    def build_graph(self, trainable=True):
        self.drop_rate = tf.placeholder(tf.float32)
        self.drop_rate_dense = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool, name="is-training")
        self.trainable = trainable

        with tf.device("cpu:0"):
            self.embedding_weights = tf.get_variable(
                name="embedding_weights",
                shape=[VOCAB_SIZE, 200],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.entity_type_weights = tf.get_variable(
                name="entity_type_weights",
                shape=[6, 4],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.pos_w0 = tf.get_variable(
                name="pos_w0",
                shape=[1, 10],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.pos_w1 = tf.get_variable(
                name="pos_w1",
                shape=[39, 10],
                dtype=tf.float32, trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.pos_weights = tf.concat([self.pos_w0, self.pos_w1], axis=0)

            self.dep_w0 = tf.get_variable(
                name="dep_w0",
                shape=[1, 10],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.dep_w1 = tf.get_variable(
                name="dep_w1",
                shape=[122, 10],
                dtype=tf.float32, trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.dep_weights = tf.concat([self.dep_w0, self.dep_w1], axis=0)

            self.distance_w0 = tf.get_variable(
                name="distance_w0",
                shape=[1, 5],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            self.distance_w1 = tf.get_variable(
                name="distance_w1",
                shape=[200, 5],
                dtype=tf.float32,
                trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.distance_weights = tf.concat([self.distance_w0,
                                               self.distance_w1], axis=0)


            self.ph_sent = tf.placeholder(tf.float32, [None, 160, 6], name='sent-input')
            self.ph_head = tf.placeholder(tf.float32, [None, 160, 6], name='head-input')

            sent_final = self.embed_feature(self.ph_sent, 'sent')
            head_final = self.embed_feature(self.ph_head, 'head')
            #combined_final = sent_final + head_final

            input_vecs = [('sent', sent_final, 3),
                          ('head', head_final, 3)]

        pools = []
        for name, context_input, window_size in input_vecs:
            input_norm = tf.layers.batch_normalization(
                context_input,
                beta_regularizer=self.regularizer,
                gamma_regularizer=self.regularizer,
                training=self.is_training,
                trainable=self.trainable,
                name='batch_norm_'+name)

            dropout_input = tf.layers.dropout(input_norm, self.drop_rate,
                                              training=self.is_training,
                                              name='dropout-'+name)

            conved = self.conv2d(name, dropout_input, window_size)
            pools.append(conved)

        # In MC-CNN, it adds up all the conv'ed outputs.
        concat_contexts = tf.add_n(pools, name='combined')

        # Max-pooling.
        conv_len = int(concat_contexts.shape[1])
        pool = tf.layers.max_pooling2d(inputs=concat_contexts,
                                       pool_size=[conv_len, 1],
                                       strides=1,
                                       padding='valid',
                                       name='final-pool')
        pool_size = self.num_kernel
        flat = tf.reshape(pool, [-1, pool_size], name="flat")

        dropout = tf.layers.dropout(flat, self.drop_rate,
                                    training=self.is_training,
                                    name='dropout-combined')

        dense1 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer,
                                 trainable=self.trainable, name='dense-1')

        dense1_batch = tf.layers.batch_normalization(
            dense1,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training,
            trainable=self.trainable,
            name='batch_norm_dense1')

        dropout1 = tf.layers.dropout(dense1_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-1')

        self.dropout1 = dropout1

        dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer,
                                 trainable=self.trainable, name='dense-2')

        dense2_batch = tf.layers.batch_normalization(
            dense2,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training,
            trainable=self.trainable,
            name='batch_norm_dense2')

        dropout2 = tf.layers.dropout(dense2_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-2')

        logits = tf.layers.dense(inputs=dropout2, units=2,
                                 kernel_regularizer=self.regularizer,
                                 trainable=self.trainable, name='output')

        self.label_placeholder = tf.placeholder(tf.int32, [None, 2])

        all_loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.label_placeholder, name='losses')
        self.loss = tf.reduce_mean(all_loss, name='batch-loss')

        self.prob = tf.nn.softmax(logits)
        pred = tf.argmax(self.prob, 1)
        gold = tf.argmax(self.label_placeholder, 1)
        tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(gold, tf.bool))
        fp = tf.logical_and(tf.cast(pred, tf.bool),
                            tf.logical_not(tf.cast(gold, tf.bool)))
        fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)),
                            tf.cast(gold, tf.bool))
        self.precision = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                                    tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)),
                                    name='precision')
        self.recall = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                                 tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)),
                                 name='recall')

        self.fscore = self.precision * self.recall * 2 / (self.precision + self.recall)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.learning_rate = tf.train.exponential_decay(
            self.lr,  # Base learning rate.
            self.global_step,  # Current index into the dataset.
            self.lr_decay_step,  # Decay step.
            self.lr_decay_rate,  # Decay rate.
            staircase=True
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            try:
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            except Exception as e:
                print(e, file=sys.stderr)


class Train(object):
    def __init__(self, model_class,folder_name='test'):
        self.model_class = model_class
        FLAGS = tf.flags.FLAGS
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.log_dir = FLAGS.log_dir
        self.save_model = FLAGS.save_model
        self.name = FLAGS.name
        self.drop_rate = FLAGS.drop_rate
        self.drop_rate_dense = FLAGS.drop_rate_dense
        self.folder_name=folder_name

    def add_summary(self):
        tf.summary.scalar('precision', self.model.precision)
        tf.summary.scalar('recall', self.model.recall)
        tf.summary.scalar('fscore', self.model.fscore)
        tf.summary.scalar('loss', self.model.loss)
        tf.summary.scalar('learning_rate', self.model.learning_rate)
        self.summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.train_dev_writer = tf.summary.FileWriter(self.log_dir + '/train_dev', self.sess.graph)
        self.dev_writer = tf.summary.FileWriter(self.log_dir + '/dev', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)

    def eval(self, name, eval_data, step):
        data = iter_sent_dataset(self.sess, eval_data, epoch=1, batch_size=10000, shuffle=False)
        sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, entity_mx, labels = next(data)

        p, r, f, l, s = self.sess.run(
            [self.model.precision, self.model.recall,
             self.model.fscore, self.model.loss, self.summary],
            feed_dict={
                self.model.ph_sent: sent_mx,
                self.model.ph_head: head_mx,
                self.model.label_placeholder: labels,
                self.model.drop_rate: 0,
                self.model.drop_rate_dense: 0,
                self.model.is_training: False,
            })
        if name == 'train':
            self.train_writer.add_summary(s, step)
        elif name == 'train_dev':
            self.train_dev_writer.add_summary(s, step)
        elif name == 'dev':
            self.dev_writer.add_summary(s, step)
        elif name == 'test':
            self.test_writer.add_summary(s, step)
        print('{}: prec {}, recall {}, fscore {}, loss {}'.format(
            name, p, r, f, l))

    def train(self, train_data, eval_sets):
        with tf.Graph().as_default():
            model = self.model_class()
            model.build_graph()
            saver = tf.train.Saver(tf.global_variables())
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            self.model = model
            self.batch_num=len(glob(train_data))
            print('File number of training data:',self.batch_num)

            with tf.Session() as sess:
                self.sess = sess
                self.add_summary()
                sess.run([init, init_l])
                sess.run(tf.assign(model.embedding_weights, EMBEDDING))
                sess.run(tf.assign(model.entity_type_weights, [[0, 0, 0, 0],
                                                               [1, 0, 0, 0],
                                                               [0, 1, 0, 0],
                                                               [0, 0, 1, 0],
                                                               [0, 0, 0, 1],
                                                               [0, 0, 1, 1]]))
                losses = []

                step = 1
                for batch_data in iter_sent_dataset(sess, train_data, self.epoch, self.batch_size):
                    (batch_sent_mx, batch_sent_len,
                     batch_head_mx, batch_head_len,
                     batch_dep_mx, batch_dep_len,
                     batch_entity, batch_labels) = batch_data
                    _, mini_loss = sess.run(
                        [model.train_op, model.loss],
                        feed_dict={
                            model.ph_sent: batch_sent_mx,
                            model.ph_head: batch_head_mx,
                            model.label_placeholder: batch_labels,
                            model.drop_rate: self.drop_rate,
                            model.drop_rate_dense: self.drop_rate_dense,
                            model.is_training: True,
                        })

                    losses.append(mini_loss)

                    if DEBUG or (step > 0 and step % self.batch_num == 0):
                        print('\n{}: step {}, loss {}'.format(
                            str(datetime.now()), step, np.mean(losses)))

                        for name, eval_data in eval_sets:
                            self.eval(name, eval_data, step)
                        losses = []
                        if self.save_model and step % (self.batch_num*10)==0:
                            step_name=str(int(step/(self.batch_num)))
                            global_step = sess.run(model.global_step)
                            path = saver.save(sess, 'model_cnn/'+self.folder_name+'/'+self.name+'_'+step_name)

                    step += 1



class CVTrain(object):
    def __init__(self, model_class):
        self.model_class = model_class
        FLAGS = tf.flags.FLAGS
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.log_dir = FLAGS.log_dir
        self.save_model = FLAGS.save_model
        self.name = FLAGS.name
        self.drop_rate = FLAGS.drop_rate
        self.drop_rate_dense = FLAGS.drop_rate_dense

    def train_fold(self, fold, train_sent, train_head, train_labels, test_sent, test_head, test_labels):
        with tf.Graph().as_default():
            model = self.model_class()
            model.build_graph()
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()

            with tf.Session() as sess:
                sess.run([init, init_l])
                sess.run(tf.assign(model.embedding_weights, EMBEDDING))
                sess.run(tf.assign(model.entity_type_weights,
                                   [[0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 1]]))

                data_size = len(train_labels)
                batch_num = data_size / 128 + 1

                for epoch in range(50):
                    shuf = np.random.permutation(np.arange(data_size))
                    train_sent = train_sent[shuf]
                    train_head = train_head[shuf]
                    train_labels = train_labels[shuf]

                    for batch in range(batch_num):
                        batch_start = batch * 128
                        batch_end = (batch+1) * 128
                        batch_end = data_size if batch_end > data_size else batch_end
                        batch_index = list(range(batch_start, batch_end))
                        batch_sent = train_sent[batch_index,:,:]
                        batch_head = train_head[batch_index,:,:]
                        batch_labels = train_labels[batch_index,:]
                        _, mini_loss = sess.run(
                            [model.train_op, model.loss],
                            feed_dict={
                                model.ph_sent: batch_sent,
                                model.ph_head: batch_head,
                                model.label_placeholder: batch_labels,
                                model.drop_rate: self.drop_rate,
                                model.drop_rate_dense: self.drop_rate_dense,
                                model.is_training: True,
                            })
                        #print('{}: fold {}, epoch {}, batch {}, loss {}'.format(
                        #    str(datetime.now()), fold, epoch, batch, mini_loss))

                p, r, f, l = sess.run(
                    [model.precision, model.recall, model.fscore, model.loss],
                    feed_dict={
                        model.ph_sent: test_sent,
                        model.ph_head: test_head,
                        model.label_placeholder: test_labels,
                        model.drop_rate: 0,
                        model.drop_rate_dense: 0,
                        model.is_training: False,
                    })

                print(p, r, f, l)

            del model
            return p, r, f

    def train(self, all_data_files):
        with tf.Session() as sess:
            all_data = iter_sent_dataset(sess, all_data_files, 1, 10000, True)
        sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, entity, labels = next(all_data)
        data_size = len(labels)
        chunk_size = data_size / 10

        cv_prec, cv_recall, cv_fscore = [], [], []
        for fold in range(10):
            test_chunk = (fold * chunk_size, (fold+1) * chunk_size)
            test_index = list(range(test_chunk[0], test_chunk[1]))
            test_sent = sent_mx[test_index, :, :]
            test_head = head_mx[test_index, :, :]
            test_labels = labels[test_index, :]

            train_index = [index for index in range(data_size) if index not in test_index]
            train_sent = sent_mx[train_index, :, :]
            train_head = head_mx[train_index, :, :]
            train_labels = labels[train_index, :]

            p, r, f = self.train_fold(fold, train_sent, train_head, train_labels, test_sent, test_head, test_labels)

            cv_prec.append(p)
            cv_recall.append(r)
            cv_fscore.append(f)

        print(sum(cv_prec)/10, sum(cv_recall)/10, sum(cv_fscore)/10)

if __name__ == '__main__':
    for folder_name in ['baseline','CP','CP_HP','HP']:

        train = Train(CNNModel,folder_name)
        train.train('data/pretrain/'+folder_name+'/'+folder_name+'_train_*.tfrecords',
                    [('dev', 'data/pretrain/'+folder_name+'/'+folder_name+'_train_dev.tfrecords')])
        #('train_dev', 'data/ds/filtered_train_dev.tfrecords'),
        #('test', 'data_attn/{}_dev_cont.tfrecords'.format(DATASETS[RELATION]))])
