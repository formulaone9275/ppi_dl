from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from collections import defaultdict
from constant import *
from PCNN_model import CNNContextModel
from glob import glob
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from build_tfrecord_data_new import load_context_matrix, load_tagged, create_tensor,load_context_matrix_v1
import random

def finetune_cont_model(train_data,dev_data,test_data,folder_name,cv_i):
    with tf.Graph().as_default():
        model = CNNContextModel()
        model.build_graph(True)
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            init_l = tf.local_variables_initializer()
            sess.run([init, init_l])
            saver.restore(sess, 'model/'+folder_name+'/pcnn_model_10')
            sess.run(tf.assign(model.global_step, 0))
            lr = sess.run(model.learning_rate)
            #print(lr)
            left_mx, middle_mx, right_mx, dep_mx, entity_mx, labels = train_data
            feed_dict = {
                model.left_placeholder: left_mx,
                model.middle_placeholder: middle_mx,
                model.right_placeholder: right_mx,
                model.dep_placeholder: dep_mx,
                model.entity_placeholder: entity_mx,
                model.drop_rate: 0,
                model.drop_rate_dense: 0,
                model.is_training: False,
                model.label_placeholder: labels,
            }
            
            p, r, f, prob = sess.run(
                [model.precision, model.recall, model.fscore, model.prob],
                feed_dict=feed_dict)
            print(p,r,f)

            data_size = len(left_mx)
            batch_size = 128
            batch_num = int(data_size / batch_size)
            dev_f_score_step=[]
            test_f_score_step=[]
            for epoch in range(200):
                shuf = np.random.permutation(np.arange(data_size))
                left_mx = left_mx[shuf]
                right_mx = right_mx[shuf]
                middle_mx = middle_mx[shuf]
                dep_mx = dep_mx[shuf]
                entity_mx = entity_mx[shuf]
                labels = labels[shuf]
                
                for batch in range(batch_num):
                    batch_start = batch * batch_size
                    batch_end = (batch + 1) * batch_size
                    if batch_end > data_size:
                        batch_end = data_size
                        
                    b_left = left_mx[batch_start:batch_end]
                    b_right = right_mx[batch_start:batch_end]
                    b_middle = middle_mx[batch_start:batch_end]
                    b_dep = dep_mx[batch_start:batch_end]
                    b_entity = entity_mx[batch_start:batch_end]
                    b_label = labels[batch_start:batch_end]

                    feed_dict={
                        model.left_placeholder: b_left,
                        model.middle_placeholder: b_middle,
                        model.right_placeholder: b_right,
                        model.dep_placeholder: b_dep,
                        model.entity_placeholder: b_entity,
                        model.is_training: True,
                        model.label_placeholder: b_label,
                        model.drop_rate_dense: tf.flags.FLAGS.drop_rate_dense,
                        model.drop_rate: tf.flags.FLAGS.drop_rate,
                    }
                    
                    _, mini_loss = sess.run([model.train_op, model.loss],
                                            feed_dict=feed_dict)
                #see the performance on development set
                if (epoch+1)%25==0:
                    left_mx_dev, middle_mx_dev, right_mx_dev, dep_mx_dev, entity_mx_dev, labels_dev = dev_data
                    feed_dict = {
                        model.left_placeholder: left_mx_dev,
                        model.middle_placeholder: middle_mx_dev,
                        model.right_placeholder: right_mx_dev,
                        model.dep_placeholder: dep_mx_dev,
                        model.entity_placeholder: entity_mx_dev,
                        model.is_training: False,
                        model.label_placeholder: labels_dev,
                        model.drop_rate_dense: 0,
                        model.drop_rate: 0
                    }

                    p, r, f, prob = sess.run(
                        [model.precision, model.recall, model.fscore, model.prob],
                        feed_dict=feed_dict)
                    print(epoch+1,'performance on dev set',p,r,f)
                    if tf.flags.FLAGS.save_model:
                        global_step = sess.run(model.global_step)
                        path = saver.save(sess, 'model/'+folder_name+'/'+tf.flags.FLAGS.name+'cv'+str(cv_i)+'_transfer_step'+str(epoch+1))
                    dev_f_score_step.append(f)


                    #see the test set performance

                    left_mx_test, middle_mx_test, right_mx_test, dep_mx_test, entity_mx_test, labels_dev = test_data
                    feed_dict = {
                        model.left_placeholder: left_mx_test,
                        model.middle_placeholder: middle_mx_test,
                        model.right_placeholder: right_mx_test,
                        model.dep_placeholder: dep_mx_test,
                        model.entity_placeholder: entity_mx_test,
                        model.is_training: False,
                        model.label_placeholder: labels_test,
                        model.drop_rate_dense: 0,
                        model.drop_rate: 0
                    }

                    p, r, f, prob = sess.run(
                        [model.precision, model.recall, model.fscore, model.prob],
                        feed_dict=feed_dict)
                    print('Performance on test set',p,r,f)
                    test_f_score_step.append(f)

            del model
            print('epoch {}, batch {}, loss {}'.format(epoch, batch, mini_loss))
            return dev_f_score_step,test_f_score_step


def get_train_test_file(filename,cv,test=True):

    files_all = glob(filename)
    if test:
        files = glob(files_all[cv])

    else:
        del files_all[cv]
        files=files_all
    return files
        
if __name__ == '__main__':

    cv=10

    file_name='./data/model_instance/fold*.txt'
    folder_name='CP'
    if tf.flags.FLAGS.model == 'pcnn':

        cv_prec, cv_recall, cv_fscore = [], [], []
        cv_fscore_dev=[]
        for ii in range(cv):
            file_to_train=get_train_test_file(file_name,ii,False)
            random.shuffle(file_to_train)

            #the first 8 files to train, the last file for development
            left_train, middle_train, right_train, dep_train,entity_train,label_train=[],[],[],[],[],[]
            for jj in range(len(file_to_train)-1):
                left, middle, right, dep, entity, labels=[],[],[],[],[],[]
                left, middle, right, dep, entity, labels = load_context_matrix_v1(file_to_train[jj])

                left_train+=left
                middle_train+=middle
                right_train+=right
                dep_train+=dep
                entity_train+=entity
                label_train+=list(labels)

            left_mx, left_len = create_tensor(left_train,20, [0] * 6)
            middle_mx, middle_len = create_tensor(middle_train,80, [0] * 6)
            right_mx, right_len = create_tensor(right_train,20, [0] * 6)
            dep_mx, dep_len = create_tensor(dep_train,20, [0] * 6)
            entity_mx = np.array(entity_train)
            train_data = [left_mx, middle_mx, right_mx, dep_mx, entity_mx, np.array(label_train)]

            #get development data
            left_dev, middle_dev, right_dev, dep_dev, entity_dev, labels_dev=[],[],[],[],[],[]
            left_dev, middle_dev, right_dev, dep_dev, entity_dev, labels_dev = load_context_matrix_v1(file_to_train[-1])

            left_mx_dev, left_len_dev = create_tensor(left_dev,20, [0] * 6)
            middle_mx_dev, middle_len_dev = create_tensor(middle_dev,80, [0] * 6)
            right_mx_dev, right_len_dev = create_tensor(right_dev,20, [0] * 6)
            dep_mx_dev, dep_len_dev = create_tensor(dep_dev,20, [0] * 6)
            entity_mx_dev = np.array(entity_dev)
            dev_data = [left_mx_dev, middle_mx_dev, right_mx_dev, dep_mx_dev, entity_mx_dev, np.array(labels_dev)]

            #get the test set
            file_to_test=get_train_test_file(file_name,ii,True)
            left_test, middle_test, right_test, dep_test, entity_test, labels_test=[],[],[],[],[],[]
            left_test, middle_test, right_test, dep_test, entity_test, labels_test = load_context_matrix_v1(file_to_test[-1])

            left_mx_test, left_len_test = create_tensor(left_test,20, [0] * 6)
            middle_mx_test, middle_len_test = create_tensor(middle_test,80, [0] * 6)
            right_mx_test, right_len_test = create_tensor(right_test,20, [0] * 6)
            dep_mx_test, dep_len_test = create_tensor(dep_test,20, [0] * 6)
            entity_mx_test = np.array(entity_test)
            test_data = [left_mx_test, middle_mx_test, right_mx_test, dep_mx_test, entity_mx_test, np.array(labels_test)]

            #train the model and
            dev_f,test_f = finetune_cont_model(train_data, dev_data,test_data,folder_name,ii)
            #cv_prec.append(p)
            #cv_recall.append(r)
            cv_fscore.append(test_f)
            cv_fscore_dev.append(dev_f)
        #print(sum(cv_prec)/len(cv_prec), sum(cv_recall)/len(cv_recall), sum(cv_fscore)/len(cv_fscore))
        with open('model/'+folder_name+'/test_fscore.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(cv_fscore, f, pickle.HIGHEST_PROTOCOL)
        with open('model/'+folder_name+'/dev_fscore.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(cv_fscore_dev, f, pickle.HIGHEST_PROTOCOL)



        

