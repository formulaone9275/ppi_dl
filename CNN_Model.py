from __future__ import print_function
#from data_utils_new import load_sentence_matrix, pad_and_prune_seq
import tensorflow as tf
import numpy as np
#from embedding_utils import VOCAB_SIZE, EMBEDDING
import sklearn as sk
from glob import glob
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
import matplotlib.pyplot as plt

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
    def __init__(self,model_index):
        self.cv=True
        self.model_index=model_index
        self.saver=tf.train.Saver(tf.global_variables())
        self.sess=tf.Session()
        self.pretrain_indicator=True
        #self.training_data_file=training_data_file
        #self.test_data_file=test_data_file

    def build(self):

        self.x = tf.placeholder(tf.float32, shape=[None, 320, 354])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 2])

        x1,x2=tf.split(self.x,num_or_size_splits=2,axis=1)
        print(x1.get_shape())
        print(x2.get_shape())
        # Convolutional Layer #11
        conv11 = tf.layers.conv2d(
            inputs=tf.expand_dims(x1,axis=3),
            filters=400,
            kernel_size=[3, 354],
            padding="valid",
            activation=tf.nn.relu)

        # Pooling Layer #11
        pool11 = tf.layers.max_pooling2d(inputs=conv11, pool_size=[158,1], strides=1)
        # Convolutional Layer #12
        conv12 = tf.layers.conv2d(
            inputs=tf.expand_dims(x2,axis=3),
            filters=400,
            kernel_size=[3, 354],
            padding="valid",
            activation=tf.nn.relu)

        #combined_conv = conv11 + conv12
        #combined = tf.nn.relu(combined_conv)


        # Pooling Layer
        pool12 = tf.layers.max_pooling2d(inputs=conv12, pool_size=[158,1], strides=1)

        pools=tf.concat([pool11,pool12],axis=1)
        print(pools.get_shape())
        # Dense Layer
        pool2_flat = tf.reshape(pools, [-1, 800])
        #
        print(pool2_flat.get_shape())

        self.keep_prob = tf.placeholder(tf.float32)
        self.IsTraining = tf.placeholder(tf.bool)
        dropout = tf.layers.dropout(
            inputs=pool2_flat, rate=self.keep_prob,training=self.IsTraining)
        dense = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(
            inputs=dense, rate=self.keep_prob,training=self.IsTraining)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout1, units=2)
        y = tf.nn.softmax(logits)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        train_step = tf.train.AdamOptimizer(7e-4).minimize(cross_entropy)
        self.y_p = tf.argmax(y, 1)
        self.y_t = tf.argmax(self.y_, 1)
        #calculate the precision, recall and F score
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        rec, rec_op = tf.metrics.recall(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        pre, pre_op = tf.metrics.precision(labels=tf.argmax(self.y_, 1), predictions=self.y_p)
        self.train_step=train_step
        self.cross_entropy=cross_entropy
    def pretrain(self):

        #train the model with the data from distant supervision
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        iteration_error=[]
        for i in range(10):

            step_error=0
            batch_num=1
            for batch_data in iter_sent_dataset_pretrain(self.sess, 'data/pretrain/aimed_pretrain*.tfrecords', 128,True):

                input_data,label_list=batch_data
                #train the model
                self.train_step.run(feed_dict={self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:True})
                #calculate the cross entropy for this small step
                ce = self.cross_entropy.eval(feed_dict={
                    self.x: input_data, self.y_: label_list, self.keep_prob: 0,self.IsTraining:False})
                if batch_num%10==0:
                    print('Epoch %d, batch %d, cross_entropy %g' % (i+1,batch_num, ce),)

                step_error+=ce
                batch_num+=1
            iteration_error.append(step_error)
            print("Epoch error:",step_error)
        print("Error change:")
        print(iteration_error)
        #save the global variable
        self.saver.save(self.sess,"model/model.ckpt")


    def train(self):

        cross_validation=10

        #with tf.Session() as sess:

        self.over_all_f_score=[]
        for c in range(cross_validation):
            print("dataset %d as the test dataest"%c)
            #initialize everything to start again
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            if self.pretrain_indicator=True:
                self.saver.restore(self.sess,"model/model.ckpt")
            #record the cross entropy each step during training
            iteration_error=[]
            for i in range(250):

                step_error=0
                batch_num=1
                for batch_data in iter_sent_dataset(self.sess, 'data/model'+str(self.model_index)+'/aimed_cross_validataion*.tfrecords', 128,True,c,False):

                    input_data,label_list=batch_data
                    #train the model
                    self.train_step.run(session=self.sess,feed_dict={self.x: input_data, self.y_: label_list, self.keep_prob: 0.5,self.IsTraining:True})
                    #calculate the cross entropy for this small step
                    ce = self.cross_entropy.eval(session=self.sess,feed_dict={
                        self.x: input_data, self.y_: label_list, self.keep_prob: 0,self.IsTraining:False})
                    #if batch_num%10==0:
                    #    print('Epoch %d, batch %d, cross_entropy %g' % (i+1,batch_num, ce),)

                    step_error+=ce
                    batch_num+=1
                iteration_error.append(step_error)
                print("Epoch %d, Cross entropy:%g"%(i+1,step_error))
            #print("Error change:")
            #print(iteration_error)
            '''
            plt.figure()
            plt.plot(range(len(iteration_error)), iteration_error,linewidth=2)
            plt.title('Loss function', fontsize=20)
            plt.xlabel('Epoch Time', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.show()
            '''
            #calculate the training F score
            y_pred_training=[]
            y_true_training=[]
            for batch_data in iter_sent_dataset(self.sess, 'data/model'+str(self.model_index)+'/aimed_cross_validataion*.tfrecords', 128,True,c,False):
                input_data_training,label_list_training=batch_data
                #one way to calculate precision recall F score
                y_pred_training+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_training, self.y_: label_list_training, self.keep_prob: 0,self.IsTraining:False}))
                y_true_training+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_training, self.y_: label_list_training, self.keep_prob: 0,self.IsTraining:False}))

            print("Accuracy of training", sk.metrics.accuracy_score(y_true_training, y_pred_training))
            print("Precision of training", sk.metrics.precision_score(y_true_training, y_pred_training))
            print("Recall of training", sk.metrics.recall_score(y_true_training, y_pred_training))
            print("f1_score of training", sk.metrics.f1_score(y_true_training, y_pred_training))


            y_pred=[]
            y_true=[]
            for batch_data in iter_sent_dataset(self.sess, 'data/model'+str(self.model_index)+'/aimed_cross_validataion*.tfrecords', 128,True,c,True):
                input_data_all_test,label_list_test=batch_data
                #one way to calculate precision recall F score
                y_pred+=list(self.y_p.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0,self.IsTraining:False}))
                y_true+=list(self.y_t.eval(session=self.sess,feed_dict={self.x: input_data_all_test, self.y_: label_list_test, self.keep_prob: 0,self.IsTraining:False}))

            print("Accuracy", sk.metrics.accuracy_score(y_true, y_pred))
            print("Precision", sk.metrics.precision_score(y_true, y_pred))
            print("Recall", sk.metrics.recall_score(y_true, y_pred))
            print("f1_score", sk.metrics.f1_score(y_true, y_pred))
            self.over_all_f_score.append(sk.metrics.f1_score(y_true, y_pred))
        print("Final F score:",sum(self.over_all_f_score)/len(self.over_all_f_score))

if __name__ == '__main__':
    for i in range(5):
        Model=CNNModel(i+1)
        Model.build()
        Model.train()
        del Model