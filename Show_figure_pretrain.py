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

def load_pickle_file(file_path,file_name):
    with open(file_path+'/'+file_name, 'rb') as f:
        pretrain_f_score = pickle.load(f,encoding="latin1")
    return pretrain_f_score

if __name__ == '__main__':
    interval=1
    for file_path in ['baseline','CP','CP_TW','filtered']:
        pretrain_f_score=load_pickle_file('model/'+file_path,'pcnn_model.pickle')
        x_axle_a=[(a+1)*interval for a in range(len(pretrain_f_score))]
        plt.figure()
        plt.plot(x_axle_a, pretrain_f_score,linewidth=2)
        plt.title('F score on dev set of '+file_path, fontsize=20)
        plt.xlabel('Epoch Time', fontsize=16)
        plt.ylabel('F Score', fontsize=16)
        plt.show()