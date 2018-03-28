from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
from constant import *
from PCNN_model import CNNContextModel

from build_tfrecord_data_new import load_context_matrix, load_tagged, create_tensor

def evaluate_cont_model(eval_data):
    #map_index = index_2_word()
    with tf.Graph().as_default():
        model = CNNContextModel()
        model.build_graph()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            saver.restore(sess, tf.flags.FLAGS.model_path)
            left, middle, right, dep, entity, label = eval_data
            feed_dict = {
                model.left_placeholder: left,
                model.middle_placeholder: middle,
                model.right_placeholder: right,
                model.dep_placeholder: dep,
                model.entity_placeholder: entity,
                model.label_placeholder: label,
                model.drop_rate: 0,
                model.drop_rate_dense: 0,
                model.is_training: False,
            }
            p, r, f, prob, l = sess.run(
                [model.precision, model.recall, model.fscore, model.prob, model.loss],
                feed_dict=feed_dict)
            print(p, r, f, l)


if __name__ == '__main__':


    if tf.flags.FLAGS.model == 'pcnn':
        test_data, entity, labels = load_context_matrix('data/aimed.txt')
        left, middle, right, dep = test_data
        left_mx, left_len = create_tensor(*left)
        middle_mx, middle_len = create_tensor(*middle)
        right_mx, right_len = create_tensor(*right)
        dep_mx, dep_len = create_tensor(*dep)
        eval_data = [left_mx, middle_mx, right_mx, dep_mx, entity, labels]
        evaluate_cont_model(eval_data)
