from __future__ import print_function
#from data_utils_new import load_sentence_matrix, pad_and_prune_seq
import tensorflow as tf
import numpy as np
#from embedding_utils import VOCAB_SIZE, EMBEDDING
import sklearn as sk
from glob import glob
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
import matplotlib.pyplot as plt
import pickle

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _parse_function(example_proto):
    features = {
        'seq': tf.FixedLenFeature([52700,], tf.float32),
        'seq_len': tf.FixedLenFeature([3,], tf.float32),
        'label': tf.FixedLenFeature([2,], tf.float32),
    }
    parsed = tf.parse_single_example(example_proto, features)
    seq = tf.reshape(parsed['seq'], [340, 155])
    sent, head, dep = tf.split(seq, [160, 160, 20], axis=0)
    sent_len, head_len, dep_len = tf.split(parsed['seq_len'], [1, 1, 1])
    return sent, sent_len, head, head_len, dep, dep_len, parsed['label']


def build_dataset(filename, target):
    data, labels = load_sentence_matrix(filename)

    sent_data, head_data, dep_data = data
    #print(sent_data[0])
    sent_mx, max_sent_len, sent_padding = sent_data
    head_mx, max_head_len, head_padding = head_data
    dep_mx, max_dep_len, dep_padding = dep_data
    input_data=[]
    label_data=[]
    #writer = tf.python_io.TFRecordWriter(target)
    for sent, head,  label in zip(sent_mx, head_mx, labels):
        sent, sent_len = pad_and_prune_seq(sent, max_sent_len, sent_padding)
        head, head_len = pad_and_prune_seq(head, max_head_len, head_padding)
        #dep, dep_len = pad_and_prune_seq(dep, max_dep_len, dep_padding)

        all_seq = np.concatenate((sent, head), axis=0)


        all_len = [sent_len, head_len]
        input_data.append(all_seq)
        label_data.append(label)
    return input_data,label_data
    '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'seq': _float_feature(np.ravel(all_seq)),
            'seq_len': _float_feature(all_len),
            'label': _float_feature(label),
        }))

        writer.write(example.SerializeToString())
    writer.close()
'''

def iter_dataset(sess, filename, epoch=None, batch_size=None):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    if epoch is not None:
        dataset = dataset.repeat(epoch)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            batch = sess.run(next_element)
            sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, labels = batch
            sent_len = np.ravel(sent_len)
            head_len = np.ravel(head_len)
            dep_len = np.ravel(dep_len)
            yield sent_mx, sent_len, head_mx, head_len, dep_mx, dep_len, labels
        except tf.errors.OutOfRangeError:
            break


def read_data(filename):
    input_data,label=build_dataset(filename, 'data/aimed_training_p.tfrecords')
    #print(np.shape(label))
    #print(label[0:100])
    #concantenate the embedding vector
    input_data_all_sen=[]
    input_data_all_head=[]
    for ii in range(len(input_data)):
        input_data_all_temp_sen=[]
        input_data_all_temp_head=[]
        for jj in range(len(input_data[0])):
            if jj<160:
                temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
                #print(temp)
                input_data_all_temp_sen.append(temp)
            else:
                temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
                #print(temp)
                input_data_all_temp_head.append(temp)

        input_data_all_sen.append(input_data_all_temp_sen)
        input_data_all_head.append(input_data_all_temp_head)

    label_list=[]
    for kk in range(len(label)):
        label_list.append(list(label[kk]))
    #label_t=tf.reshape(label_list,[len(label),2])
    #print(input_data_all.get_shape())
    return input_data_all_sen,input_data_all_head,label_list

def read_data_v2(filename):
    input_data,label=build_dataset(filename, 'data/aimed_training_p.tfrecords')
    #print(np.shape(label))
    #print(label[0:100])
    #concantenate the embedding vector
    input_data_all=[]

    for ii in range(len(input_data)):
        input_data_all_temp=[]

        for jj in range(len(input_data[0])):
            temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
            input_data_all_temp.append(temp)

        input_data_all.append(input_data_all_temp)


    label_list=[]
    for kk in range(len(label)):
        label_list.append(list(label[kk]))
    #label_t=tf.reshape(label_list,[len(label),2])
    #print(input_data_all.get_shape())
    return input_data_all,label_list


def build_tfrecord_data(filename,target_filename):
    input_data,label=build_dataset(filename, target_filename)
    #print(np.shape(label))
    #print(label[0:100])
    #concantenate the embedding vector
    input_data_all=[]
    label_list=[]
    writer = tf.python_io.TFRecordWriter(target_filename)
    for ii in range(len(input_data)):
        input_data_all_temp=[]

        for jj in range(len(input_data[0])):
            temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
            input_data_all_temp.append(temp)

        input_data_all.append(input_data_all_temp)
        label_list.append(list(label[ii]))

        example = tf.train.Example(features=tf.train.Features(feature={
            'data': _float_feature(np.ravel(input_data_all_temp)),
            'label': _float_feature(label[ii]),
        }))
        writer.write(example.SerializeToString())
    writer.close()



    #label_t=tf.reshape(label_list,[len(label),2])
    #print(input_data_all.get_shape())
    #writer = tf.python_io.TFRecordWriter(target_filename)

    return input_data_all,label_list

def _sent_parse_func(example_proto):
    features = {
        'data': tf.FixedLenFeature([113280,], tf.float32),
        'label': tf.FixedLenFeature([2,], tf.float32),
    }
    parsed = tf.parse_single_example(example_proto, features)
    seq = tf.reshape(parsed['data'], [320, 354])
    sent = tf.cast(seq,tf.float32)
    labels = tf.cast(parsed['label'],tf.float32)
    return sent, labels


def iter_sent_dataset(sess, filename,  batch_size, shuffle=True,cv=0,test=True):
    ph_files = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(ph_files)
    dataset = dataset.map(_sent_parse_func, num_parallel_calls=8)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    files_all = glob(filename)
    if test:
        files = glob(files_all[cv])
        #print(files)
        #print(files)
    else:
        del files_all[cv]
        files=files_all
        #print(files)


    #random.shuffle(files)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer,{ph_files: files})

    while True:
        try:
            batch = sess.run(next_element)
            #sess.run(ph_files,{ph_files: files})
            sent_mx, labels = batch
            yield sent_mx, labels
        except tf.errors.OutOfRangeError:
            break

def iter_sent_dataset_pretrain(sess, filename,  batch_size, shuffle=True):
    ph_files = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(ph_files)
    dataset = dataset.map(_sent_parse_func, num_parallel_calls=8)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    files_all = glob(filename)

    #random.shuffle(files)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    sess.run(iterator.initializer,{ph_files: files_all})

    while True:
        try:
            batch = sess.run(next_element)
            #sess.run(ph_files,{ph_files: files})
            sent_mx, labels = batch
            yield sent_mx, labels
        except tf.errors.OutOfRangeError:
            break

class CNNModel(object):

    def __init__(self,model_index,ckpt_file_path,tfrecords_file_path,tfrecords_file_path_pretrain,pickle_file_path,ckpt_file,folder_name,tfrecords_file_path_gang):
        self.cv=10
        self.model_index=model_index
        self.ckpt_file_path=ckpt_file_path
        self.tfrecords_file_path=tfrecords_file_path
        self.tfrecords_file_path_gang=tfrecords_file_path_gang
        self.tfrecords_file_path_pretrain=tfrecords_file_path_pretrain
        self.sess=tf.Session()
        self.pretrained_indicator=False
        self.cv_indicator=False
        self.pretrain_f1_score=[]
        self.pretrain_f1_score_gang=[]
        self.pickle_file_path=pickle_file_path
        self.ckpt_file=ckpt_file
        self.pretrain_epoch=20
        self.pretrain_epoch_start=100
        self.train_epoch=200
        self.folder_name=folder_name
        self.lr=3e-3  # Base learning rate.
        self.lr_decay_step=400  # Decay step.
        self.lr_decay_rate=0.95  # Decay rate

        #self.training_data_file=training_data_file
        #self.test_data_file=test_data_file

    def build(self):

        self.x = tf.placeholder(tf.float32, shape=[None, 320, 354])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])
        self.regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4)
        self.keep_prob = tf.placeholder(tf.float32)
        self.keep_prob_dense = tf.placeholder(tf.float32)
        self.IsTraining = tf.placeholder(tf.bool)
        x1,x2=tf.split(self.x,num_or_size_splits=2,axis=1)
        #print(x1.get_shape())
        #print(x2.get_shape())
        # Convolutional Layer #11
        batch_norm_input1 = tf.layers.batch_normalization(
            tf.expand_dims(x1,axis=3),
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)
        dropout_input1 = tf.layers.dropout(
            inputs=batch_norm_input1, rate=self.keep_prob,training=self.IsTraining)
        conv11 = tf.layers.conv2d(
            inputs=dropout_input1,
            filters=400,
            kernel_size=[3, 354],
            padding="valid",
            activation=tf.nn.relu,
            kernel_regularizer =self.regularizer)

        batch_norm11 = tf.layers.batch_normalization(
            conv11,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)

        # Pooling Layer #11
        #pool11 = tf.layers.max_pooling2d(inputs=batch_norm11, pool_size=[158,1], strides=1)

        batch_norm_input2 = tf.layers.batch_normalization(
            tf.expand_dims(x2,axis=3),
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)
        dropout_input2 = tf.layers.dropout(
            inputs=batch_norm_input2, rate=self.keep_prob,training=self.IsTraining)

        # Convolutional Layer #12
        conv12 = tf.layers.conv2d(
            inputs=dropout_input2,
            filters=400,
            kernel_size=[3, 354],
            padding="valid",
            activation=tf.nn.relu,
            kernel_regularizer =self.regularizer)
        batch_norm12 = tf.layers.batch_normalization(
            conv12,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)
        #combined_conv = conv11 + conv12
        #combined = tf.nn.relu(combined_conv)
        pool11_12=tf.add_n([batch_norm11,batch_norm12])
        print(pool11_12.get_shape())
        # Pooling Layer
        pool1_2 = tf.layers.max_pooling2d(inputs=pool11_12, pool_size=[158,1], strides=1)

        #pools=tf.concat([pool11,pool12],axis=1)
        #print(pools.get_shape())
        # Dense Layer
        pool2_flat = tf.reshape(pool1_2, [-1, 400])
        #
        print(pool2_flat.get_shape())


        dropout = tf.layers.dropout(
            inputs=pool2_flat, rate=self.keep_prob,training=self.IsTraining)

        dense1 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu,
                                kernel_regularizer =self.regularizer)
        dense1_batch = tf.layers.batch_normalization(
            dense1,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)

        dropout1 = tf.layers.dropout(
            inputs=dense1_batch, rate=self.keep_prob_dense,training=self.IsTraining)

        dense2 = tf.layers.dense(inputs=dropout1, units=256, activation=tf.nn.relu,
                                 kernel_regularizer =self.regularizer)
        dense2_batch = tf.layers.batch_normalization(
            dense2,
            beta_regularizer=self.regularizer,
            gamma_regularizer=self.regularizer,
            training=self.IsTraining)

        dropout2 = tf.layers.dropout(
            inputs=dense2_batch, rate=self.keep_prob_dense,training=self.IsTraining)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout2, units=2)
        y = tf.nn.softmax(logits)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))

        #train_step = tf.train.AdamOptimizer(7e-4).minimize(cross_entropy)
        self.y_p = tf.argmax(y, 1)
        self.y_t = tf.argmax(self.y_, 1)
        #calculate the precision, recall and F score
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        rec, rec_op = tf.metrics.recall(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        pre, pre_op = tf.metrics.precision(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        #self.train_step=train_step
        self.cross_entropy=cross_entropy
        self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=None)

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
                self.train_step = optimizer.minimize(self.cross_entropy, global_step=self.global_step)
            except Exception as e:
                print(e)


    def pretrain(self):
        self.build()
        #train the model with the data from distant supervision
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        iteration_error=[]
        for i in range(self.pretrain_epoch):
            #global_step = self.sess.run(self.global_step)
            #print("global_step",global_step)
            step_error=0
            batch_num=1
            for batch_data in iter_sent_dataset_pretrain(self.sess, self.tfrecords_file_path_pretrain+"aimed_pretrain*.tfrecords", 128,False):

                input_data,label_list=batch_data
                #train the model
                self.train_step.run(session=self.sess,feed_dict={self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:True,self.keep_prob_dense:0.2})
                #calculate the cross entropy for this small step
                ce = self.cross_entropy.eval(session=self.sess,feed_dict={
                    self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2})
                if batch_num%100==0:
                    print('Epoch %d, batch %d, cross_entropy %g' % (i+1,batch_num, ce),)

                step_error+=ce
                batch_num+=1
            iteration_error.append(step_error)
            print("Epoch %d error: %g" %(i+1,step_error))
            #save the global variable
            if (i+1)%5==0:
                self.saver.save(self.sess,self.ckpt_file_path+"model_pretrain_step"+str(i+1)+".ckpt")
                self.pretrain_test()
                self.pretrain_test_gang()

        print("Error change:",iteration_error)
        #print(iteration_error)
        with open(self.pickle_file_path+'f_score_pretrain.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.pretrain_f1_score, f, pickle.HIGHEST_PROTOCOL)
        with open(self.pickle_file_path+'f_score_pretrain_gang.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.pretrain_f1_score_gang, f, pickle.HIGHEST_PROTOCOL)

        tf.Session.close(self.sess)
        tf.reset_default_graph()

    def pretrain_continue(self):
        self.build()
        #train the model with the data from distant supervision
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        print(self.ckpt_file_path)
        self.saver.restore(self.sess,self.ckpt_file_path+self.ckpt_file)
        iteration_error=[]
        for i in range(self.pretrain_epoch_start,self.pretrain_epoch):

            step_error=0
            batch_num=1
            for batch_data in iter_sent_dataset_pretrain(self.sess, self.tfrecords_file_path_pretrain+"aimed_pretrain*.tfrecords", 128,False):

                input_data,label_list=batch_data
                #train the model
                self.train_step.run(session=self.sess,feed_dict={self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:True,self.keep_prob_dense:0.2})
                #calculate the cross entropy for this small step
                ce = self.cross_entropy.eval(session=self.sess,feed_dict={
                    self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2})
                if batch_num%10==0:
                    print('Epoch %d, batch %d, cross_entropy %g' % (i+1,batch_num, ce),)

                step_error+=ce
                batch_num+=1
            iteration_error.append(step_error)
            print("Epoch %d error: %g" %(i+1,step_error))
            #save the global variable
            if (i+1)%10==0:
                self.saver.save(self.sess,self.ckpt_file_path+"model_pretrain_step"+str(i+1)+".ckpt")
                self.pretrain_test()
                self.pretrain_test_gang()

        print("Error change:")
        print(iteration_error)
        with open(self.pickle_file_path+'f_score_pretrain_continue.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.pretrain_f1_score, f, pickle.HIGHEST_PROTOCOL)


    def train(self):
        self.build()
        cross_validation=10

        #with tf.Session() as sess:
        #global_step = self.sess.run(self.global_step)
        #print("global_step",global_step)


        self.final_f_score=[]
        self.train_f_score=[]
        #make sure the pickle name are different for pretrained and not pretrained
        pickle_file_pretrain_indicator=''
        for c in range(cross_validation):
            print("dataset %d as the test dataest"%c)
            #initialize everything to start again
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            if self.pretrained_indicator==True:
                print("Will load checkpoint file!")
                pickle_file_pretrain_indicator='pretrained'
                self.saver.restore(self.sess,self.ckpt_file_path+self.ckpt_file)
            #make the global step go back to 0
            assign_step_op=self.global_step.assign(0)
            self.sess.run(assign_step_op)
            print("Global step:",self.global_step.eval())
            #record the steps f score
            self.train_f_score.append([])
            #record the cross entropy each step during training
            iteration_error=[]
            for i in range(self.train_epoch):

                step_error=0
                batch_num=1
                for batch_data in iter_sent_dataset(self.sess, self.tfrecords_file_path+"aimed_cross_validataion*.tfrecords", 128,False,c,False):

                    input_data,label_list=batch_data
                    #train the model
                    self.train_step.run(session=self.sess,feed_dict={self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:True,self.keep_prob_dense:0.2})
                    #calculate the cross entropy for this small step
                    ce = self.cross_entropy.eval(session=self.sess,feed_dict={
                        self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2})
                    #if batch_num%10==0:
                    #    print('Epoch %d, batch %d, cross_entropy %g' % (i+1,batch_num, ce),)

                    step_error+=ce
                    batch_num+=1
                iteration_error.append(step_error)
                print("Epoch %d, Cross entropy:%g"%(i+1,step_error))

                if (i+1)%25==0:
                    self.saver.save(self.sess,self.ckpt_file_path+'model_'+pickle_file_pretrain_indicator+'_train_step'+str(i+1)+'_fold'+str(c+1)+'.ckpt')
                    self.test(c)

            #calculate the training F score
            y_pred_training=[]
            y_true_training=[]
            for batch_data in iter_sent_dataset(self.sess, self.tfrecords_file_path+"aimed_cross_validataion*.tfrecords", 128,False,c,False):
                input_data_training,label_list_training=batch_data
                #one way to calculate precision recall F score
                y_pred_training+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_training, self.y_: label_list_training, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                y_true_training+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_training, self.y_: label_list_training, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))

            print("Accuracy of training", sk.metrics.accuracy_score(y_true_training, y_pred_training))
            print("Precision of training", sk.metrics.precision_score(y_true_training, y_pred_training))
            print("Recall of training", sk.metrics.recall_score(y_true_training, y_pred_training))
            print("f1_score of training", sk.metrics.f1_score(y_true_training, y_pred_training))


        '''
        print(self.f_score_steps)
        print("F score trend every 10 steps:")
        temp_f_score_steps=[]
        for jj in range(len(self.f_score_steps[0])):
            temp=0
            for ii in range(len(self.f_score_steps)):
                temp+=self.f_score_steps[ii][jj]
            temp_f_score_steps.append(temp/len(self.f_score_steps))
        print(temp_f_score_steps)
        '''
        with open(self.pickle_file_path+'f_score_steps_'+pickle_file_pretrain_indicator+'_training.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.train_f_score, f, pickle.HIGHEST_PROTOCOL)

        tf.Session.close(self.sess)
        tf.reset_default_graph()

    def pretrain_test(self):
        #with open('./data/pickle_file/f_score_training.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        #    data = pickle.load(f)
        #initialize everything to start again
        #

        if self.pretrained_indicator==True:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            print(self.ckpt_file_path)
            self.saver.restore(self.sess,self.ckpt_file_path+ckpt_file)

        if self.cv==0:
            y_pred=[]
            y_true=[]
            for batch_data in iter_sent_dataset_pretrain(self.sess, self.tfrecords_file_path+"aimed_cross_validataion*.tfrecords", 128,False):
                input_data_all_test,label_list_test=batch_data
                #one way to calculate precision recall F score
                y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
            #print("y_pred:",y_pred)
            #print("y_true:",y_true)
            #print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
            print("Precision", sk.metrics.precision_score(y_true, y_pred))
            print("Recall", sk.metrics.recall_score(y_true, y_pred))
            print("f1_score", sk.metrics.f1_score(y_true, y_pred))
            self.pretrain_f1_score.append(sk.metrics.f1_score(y_true, y_pred))
        else:
            cv_f_score=[]
            for ii in range(self.cv):
                y_pred=[]
                y_true=[]
                for batch_data in iter_sent_dataset(self.sess, self.tfrecords_file_path+"aimed_cross_validataion*.tfrecords", 128,False,ii,True):
                    input_data_all_test,label_list_test=batch_data
                    #one way to calculate precision recall F score
                    y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                    y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                #print("y_pred:",y_pred)
                #print("y_true:",y_true)
                #print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
                print("Precision", sk.metrics.precision_score(y_true, y_pred))
                print("Recall", sk.metrics.recall_score(y_true, y_pred))
                print("f1_score", sk.metrics.f1_score(y_true, y_pred))
                cv_f_score.append(sk.metrics.f1_score(y_true, y_pred))
            print("Overall F score:",sum(cv_f_score)/len(cv_f_score))
            self.pretrain_f1_score.append(cv_f_score)

    def pretrain_test_gang(self):
        #with open('./data/pickle_file/f_score_training.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        #    data = pickle.load(f)
        #initialize everything to start again
        #

        if self.pretrained_indicator==True:
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            print(self.ckpt_file_path)
            self.saver.restore(self.sess,self.ckpt_file_path+ckpt_file)

        if self.cv==0:
            y_pred=[]
            y_true=[]
            for batch_data in iter_sent_dataset_pretrain(self.sess, self.tfrecords_file_path_gang+"aimed_cross_validataion*.tfrecords", 128,False):
                input_data_all_test,label_list_test=batch_data
                #one way to calculate precision recall F score
                y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
            #print("y_pred:",y_pred)
            #print("y_true:",y_true)
            #print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
            print("Precision", sk.metrics.precision_score(y_true, y_pred))
            print("Recall", sk.metrics.recall_score(y_true, y_pred))
            print("f1_score", sk.metrics.f1_score(y_true, y_pred))
            self.pretrain_f1_score_gang.append(sk.metrics.f1_score(y_true, y_pred))
        else:
            cv_f_score=[]
            for ii in range(self.cv):
                y_pred=[]
                y_true=[]
                for batch_data in iter_sent_dataset(self.sess, self.tfrecords_file_path_gang+"aimed_cross_validataion*.tfrecords", 128,False,ii,True):
                    input_data_all_test,label_list_test=batch_data
                    #one way to calculate precision recall F score
                    y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                    y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
                #print("y_pred:",y_pred)
                #print("y_true:",y_true)
                #print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
                print("Precision of gang", sk.metrics.precision_score(y_true, y_pred))
                print("Recall of gang", sk.metrics.recall_score(y_true, y_pred))
                print("f1_score of gang", sk.metrics.f1_score(y_true, y_pred))
                cv_f_score.append(sk.metrics.f1_score(y_true, y_pred))
            print("Overall F score of gang:",sum(cv_f_score)/len(cv_f_score))
            self.pretrain_f1_score_gang.append(cv_f_score)

    def test(self,c):
        y_pred=[]
        y_true=[]
        for batch_data in iter_sent_dataset(self.sess, self.tfrecords_file_path+"aimed_cross_validataion*.tfrecords", 128,False,c,True):
            input_data_all_test,label_list_test=batch_data
            #one way to calculate precision recall F score
            y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
            y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0.5,self.IsTraining:False,self.keep_prob_dense:0.2}))
        #print("y_pred:",y_pred)
        #print("y_true:",y_true)
        #print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
        print("Precision", sk.metrics.precision_score(y_true, y_pred))
        print("Recall", sk.metrics.recall_score(y_true, y_pred))
        print("f1_score", sk.metrics.f1_score(y_true, y_pred))
        self.train_f_score[c].append(sk.metrics.f1_score(y_true, y_pred))

    def show_pretrain_figure(self):
        plt.figure()
        for ii in range(self.cv):
            fold_f_score=[]
            for jj in range(len(self.pretrain_f1_score)):
                fold_f_score.append(self.pretrain_f1_score[jj][ii])

            plt.plot(range(len(fold_f_score)), fold_f_score,linewidth=2)
        plt.title('F score change of fold '+self.folder_name, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()
        over_all_f_score=[]
        for jjk in range(len(self.pretrain_f1_score)):
            over_all_f_score.append(sum(self.pretrain_f1_score[jjk])/len(self.pretrain_f1_score[jjk]))

        plt.figure()
        plt.plot(range(len(over_all_f_score)), over_all_f_score,linewidth=2)
        plt.title('Overall F score of '+self.folder_name, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()

    def show_train_figure(self):
        plt.figure()
        over_all_f_score=[]
        for ii in range(len(self.train_f_score)):
            fold_f_score=[]
            for jj in range(self.cv):
                fold_f_score.append(self.train_f_score[jj][ii])
            over_all_f_score.append(sum(fold_f_score)/len(fold_f_score))

        for jjk in range(len(self.train_f_score)):
            x_axle=[(a+1)*10 for a in range(len(self.train_f_score[jjk]))]
            plt.plot(x_axle, self.train_f_score[jjk],linewidth=2)
        plt.title('F score change of folds '+self.folder_name, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()


        x_axle_a=[(a+1)*10 for a in range(len(over_all_f_score))]
        plt.figure()
        plt.plot(x_axle_a, over_all_f_score,linewidth=2)
        plt.title('Overall F score '+self.folder_name, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()

    def load_pickle(self):

        with open(self.pickle_file_path+'f_score_steps_training.pickle', 'rb') as f:
            self.train_f_score = pickle.load(f,encoding="latin1")

if __name__ == '__main__':

    for folder_name in ['CP','CP_TW']:
        print(folder_name)
        ckpt_file="model_pretrain_step20.ckpt"
        ckpt_file_path="./model_new/"+folder_name+"/"
        tfrecords_file_path="data/model5/"
        tfrecords_file_path_gang="data/model6/"
        tfrecords_file_path_pretrain="data/pretrain/"+folder_name+"/"
        pickle_file_path="./data/pickle_file_new/"+folder_name+"/"
        for i in range(4,5):
            print("Model "+str(i+1))
            Model=CNNModel(i+1,ckpt_file_path,tfrecords_file_path,tfrecords_file_path_pretrain,pickle_file_path,ckpt_file,folder_name,tfrecords_file_path_gang)
            Model.pretrain()
            #Model.load_pickle()
            #Model.show_pretrain_figure()

            #Model.show_pretrain_figure()
            del Model

