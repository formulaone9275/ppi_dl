from __future__ import print_function,division
import tensorflow as tf
import numpy as np
from datetime import datetime
from constant import *
from flags import *
from word_embedding import VOCAB_SIZE,EMBEDDING
from glob import glob
import tqdm
import os,random
import pickle

#from build_tfrecord_data_new import load_context_matrix, create_tensor
#from collections import defaultdict
import math
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

# context PCNN dataset.
def _cont_parse_func(example_proto):
    features = {
        'seq': tf.FixedLenFeature([140*6,], tf.float32),
        'entity': tf.FixedLenFeature([2*6,], tf.float32),
        'label': tf.FixedLenFeature([2,], tf.float32),
    }
    parsed = tf.parse_single_example(example_proto, features)
    seq = tf.reshape(parsed['seq'], [140, 6])
    seq = post_process_seq(seq)
    entity = tf.reshape(parsed['entity'], [2, 6])
    entity = post_process_seq(entity)

    left, middle, right, dep = tf.split(seq, [20, 80, 20, 20], axis=0)
    return left, middle, right, dep, entity, parsed['label']


def iter_cont_dataset(sess, filename, epoch, batch_size, shuffle=True):
    ph_files = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(ph_files)
    dataset = dataset.map(_cont_parse_func, num_parallel_calls=8)
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
                yield batch
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


class CNNContextModel(object):
    def __init__(self):
        FLAGS = tf.flags.FLAGS
        self.emb_dim = FLAGS.emb_dim
        self.num_kernel = FLAGS.num_kernel
        self.min_window = FLAGS.min_window
        self.max_window = FLAGS.max_window
        self.l2_reg = FLAGS.l2_reg
        self.lr = FLAGS.lr
        self.lr_decay_step = FLAGS.decay_step
        self.lr_decay_rate = FLAGS.decay_rate
        self.use_dep = FLAGS.use_dep
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg)

    def conv2d(self, name, inputs, feature_dim, window_stride):
        inputs = tf.expand_dims(inputs, -1)
        flats = []
        for win_size, strides in window_stride:
            conv = tf.layers.conv2d(inputs=inputs,
                                    filters=self.num_kernel,
                                    kernel_size=[win_size, feature_dim],
                                    strides=strides,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=self.regularizer,
                                    trainable=self.trainable,
                                    padding='valid',
                                    name='conv-{}-{}'.format(name, win_size))

            batch_norm = tf.layers.batch_normalization(
                conv,
                beta_regularizer=self.regularizer,
                gamma_regularizer=self.regularizer,
                training=self.is_training,
                trainable=self.trainable,
                name='batch-norm-{}-{}'.format(name, win_size))

            conv_len = int(conv.shape[1])
            pool = tf.layers.max_pooling2d(inputs=batch_norm,
                                           pool_size=[conv_len, 1],
                                           strides=1,
                                           padding='valid',
                                           name='pool-{}-{}'.format(name, win_size))
            pool_size = self.num_kernel
            flats.append(tf.reshape(pool, [-1, pool_size], name='flat-{}-{}'.format(name, win_size)))
        return flats

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

        if name == 'dep':
            embed_toe1 = tf.nn.embedding_lookup(self.dep_distance_weights, toe1)
            embed_toe2 = tf.nn.embedding_lookup(self.dep_distance_weights, toe2)
        else:
            embed_toe1 = tf.nn.embedding_lookup(self.distance_weights, toe1)
            embed_toe2 = tf.nn.embedding_lookup(self.distance_weights, toe2)

        embed_dep = tf.nn.embedding_lookup(self.dep_weights, dep)

        if name == 'dep':
            final = tf.concat([embed_token, embed_pos,
                               embed_enttype,
                               embed_toe1, embed_toe2,
                               embed_dep],
                              axis=2, name=name+'-final')
        else:
            final = tf.concat([embed_token, embed_pos,
                               embed_enttype,
                               embed_toe1, embed_toe2,
                               embed_dep],
                              axis=2, name=name+'-final')
        #print(final.shape)
        return final

    def build_graph(self, trainable=True):
        self.trainable = trainable

        with tf.device("cpu:0"):
            self.drop_rate = tf.placeholder(tf.float32)
            self.drop_rate_dense = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool, name="is-training")

            self.embedding_weights = tf.get_variable(
                name="pretrained_embedding",
                shape=[VOCAB_SIZE, 200],
                dtype=tf.float32,
                trainable=False,
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
                regularizer=self.regularizer,
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
                regularizer=self.regularizer,
                dtype=tf.float32, trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.dep_weights = tf.concat([self.dep_w0, self.dep_w1], axis=0)

            self.distance_w0 = tf.get_variable(
                name="distance_w0",
                shape=[1, 5],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            # Some of the weights won't be used due to distance limit.
            self.distance_w1 = tf.get_variable(
                name="distance_w1",
                shape=[200, 5],
                regularizer=self.regularizer,
                dtype=tf.float32, trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.distance_weights = tf.concat([self.distance_w0, self.distance_w1], axis=0)

            self.dep_distance_w0 = tf.get_variable(
                name="dep_distance_w0",
                shape=[1, 5],
                dtype=tf.float32, trainable=False,
                initializer=tf.zeros_initializer())

            # Some of the weights won't be used due to distance limit.
            self.dep_distance_w1 = tf.get_variable(
                name="dep_distance_w1",
                shape=[200, 5],
                regularizer=self.regularizer,
                dtype=tf.float32, trainable=self.trainable,
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.dep_distance_weights = tf.concat(
                [self.dep_distance_w0, self.dep_distance_w1], axis=0)


            self.left_placeholder = tf.placeholder(tf.float32, [None, 20, 6], name='left-input')
            self.middle_placeholder = tf.placeholder(tf.float32, [None, 80, 6], name='middle-input')
            self.right_placeholder = tf.placeholder(tf.float32, [None, 20, 6], name='right-input')
            self.dep_placeholder = tf.placeholder(tf.float32, [None, 20, 6], name='dep-input')
            self.entity_placeholder = tf.placeholder(tf.float32, [None, 2, 6], name='entity-input')

            input_vecs = []
            for name, placeholder, window_stride in [
                ('left', self.left_placeholder, [(3, 1),]),
                ('middle', self.middle_placeholder, [(3, 1), ]),
                ('right', self.right_placeholder, [(3, 1), ]),
                ('dep', self.dep_placeholder, [(2, 1), ]),
            ]:
                if not self.use_dep and name == 'dep':
                    continue
                input_final = self.embed_feature(placeholder, name)
                length = int(placeholder.shape[1])
                feature_dim = int(input_final.shape[-1])
                input_vecs.append((name, input_final, length, feature_dim, window_stride))

            embed_entity = []
            final = self.embed_feature(self.entity_placeholder, name='entity')
            embed_entity = tf.split(final, [1, 1], axis=1)

        pools = []
        for index, input_vec in enumerate(input_vecs):
            name, context_input, length, feature_dim, window_size = input_vec
            input_norm = tf.layers.batch_normalization(
                context_input,
                beta_regularizer=self.regularizer,
                gamma_regularizer=self.regularizer,
                training=self.is_training,
                trainable=self.trainable,
                name='batch_norm_'+name)

            dropout_input = tf.layers.dropout(input_norm, self.drop_rate,
                                              training=self.is_training, name='dropout-'+name)

            conveds = self.conv2d(name, dropout_input, feature_dim, window_size)
            pools += conveds

        concat_contexts = tf.concat(pools, axis=1, name='combined')

        dropout = tf.layers.dropout(concat_contexts, self.drop_rate,
                                    training=self.is_training, name='dropout-combined')
        self.dropout = dropout

        dense1 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu,
                                 kernel_regularizer=self.regularizer,
                                 trainable=self.trainable,
                                 name='dense-1')

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
                                 trainable=self.trainable,
                                 name='dense-2')

        dense2_batch = tf.layers.batch_normalization(
            dense2,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.is_training,
            trainable=self.trainable
        )

        dropout2 = tf.layers.dropout(dense2_batch, self.drop_rate_dense,
                                     training=self.is_training, name='dropout-2')

        logits = tf.layers.dense(inputs=dropout2, units=2,
                                 kernel_regularizer=self.regularizer,
                                 trainable=self.trainable,
                                 name='output')

        self.label_placeholder = tf.placeholder(tf.int32, [None, 2])

        all_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=self.label_placeholder,
                                                           name='losses')
        self.loss = tf.reduce_mean(all_loss, name='batch-loss')

        self.prob = tf.nn.softmax(logits)
        self.pred = tf.argmax(self.prob, 1)
        self.gold = tf.argmax(self.label_placeholder, 1)
        tp = tf.logical_and(tf.cast(self.pred, tf.bool), tf.cast(self.gold, tf.bool))
        fp = tf.logical_and(tf.cast(self.pred, tf.bool),
                            tf.logical_not(tf.cast(self.gold, tf.bool)))
        fn = tf.logical_and(tf.logical_not(tf.cast(self.pred, tf.bool)),
                            tf.cast(self.gold, tf.bool))
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
        try:
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        except:
            pass


class Train(object):
    def __init__(self,folder_name='test'):
        FLAGS = tf.app.flags.FLAGS
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.epoch = FLAGS.epoch
        self.drop_rate = FLAGS.drop_rate
        self.drop_rate_dense = FLAGS.drop_rate_dense
        self.save_model = FLAGS.save_model
        self.name = FLAGS.name
        self.dev_f_score=[]
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
        data = iter_cont_dataset(self.sess, eval_data, epoch=1, batch_size=10000, shuffle=False)
        left_mx, middle_mx, right_mx, dep_mx, entity_mx, labels = next(data)
        p, r, f, l, s = self.sess.run(
            [self.model.precision, self.model.recall,
             self.model.fscore, self.model.loss, self.summary],
            feed_dict={
                self.model.left_placeholder: left_mx,
                self.model.middle_placeholder: middle_mx,
                self.model.right_placeholder: right_mx,
                self.model.dep_placeholder: dep_mx,
                self.model.entity_placeholder: entity_mx,
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
        self.dev_f_score.append(f)

    def train(self, train_data, eval_sets):
        with tf.Graph().as_default():
            model = CNNContextModel()
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
                sess.run(init)
                sess.run(init_l)
                sess.run(tf.assign(model.embedding_weights, EMBEDDING))
                sess.run(tf.assign(model.entity_type_weights, [[0, 0, 0, 0],
                                                               [1, 0, 0, 0],
                                                               [0, 1, 0, 0],
                                                               [0, 0, 1, 0],
                                                               [0, 0, 0, 1],
                                                               [0, 0, 1, 1]]))
                losses = []
                step = 1
                for batch_data in iter_cont_dataset(sess, train_data, self.epoch, self.batch_size):
                    (batch_left_mx, batch_middle_mx, batch_right_mx, batch_dep_mx,
                     batch_entity_mx, batch_labels) = batch_data

                    _, mini_loss = sess.run(
                        [model.train_op, model.loss],
                        feed_dict={
                            model.left_placeholder: batch_left_mx,
                            model.middle_placeholder: batch_middle_mx,
                            model.right_placeholder: batch_right_mx,
                            model.dep_placeholder: batch_dep_mx,
                            model.entity_placeholder: batch_entity_mx,
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
                            path = saver.save(sess, 'model/'+self.folder_name+'/'+self.name+'_'+step_name)

                    step += 1


                #save the f score of development set to choose epoch time
                with open('model/'+self.folder_name+'/'+self.name+'.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(self.dev_f_score, f, pickle.HIGHEST_PROTOCOL)
                print(self.dev_f_score)
    '''
    def show_train_figure(self):
        plt.figure()

        x_axle=[(a+1) for a in range(len(self.dev_f_score))]
        plt.plot(x_axle, self.dev_f_score,linewidth=2)
        plt.title('F score change of dev data of '+self.folder_name, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()
    '''
if __name__ == '__main__':

    # distant data.
    '''
    for folder_name in ['baseline','CP','CP_TW','filtered']:
        if tf.flags.FLAGS.model == 'pcnn':
            train = Train(folder_name)
            train.train('data/ds/'+folder_name+'/'+folder_name+'_train_*.tfrecords',
                        [('dev', 'data/ds/'+folder_name+'_dev.tfrecords')])
            #('train_dev', 'data/ds/filtered_train_dev.tfrecords'),
            #('test', 'data_attn/{}_dev_cont.tfrecords'.format(DATASETS[RELATION]))])
    '''
    for folder_name in ['baseline','CP','CP_HP','HP']:
        if tf.flags.FLAGS.model == 'pcnn':
            train = Train(folder_name)
            train.train('data/ds/'+folder_name+'/'+folder_name+'_train_*.tfrecords',
                        [('dev', 'data/ds/'+folder_name+'_train_dev.tfrecords')])

