from __future__ import print_function

import matplotlib.pyplot as plt
import pickle

def show_train_figure(folder_name,train_f_score,interval):
    cv=10
    plt.figure()
    over_all_f_score=[]
    for ii in range(len(train_f_score[0])):
        fold_f_score=[]
        for jj in range(len(train_f_score)):
            fold_f_score.append(train_f_score[jj][ii])
        over_all_f_score.append(sum(fold_f_score)/len(fold_f_score))

    for jjk in range(len(train_f_score)):
        x_axle=[(a+1)*interval for a in range(len(train_f_score[jjk]))]
        plt.plot(x_axle, train_f_score[jjk],linewidth=2)
    plt.title('F score change of folds '+folder_name, fontsize=20)
    plt.xlabel('Epoch Time', fontsize=16)
    plt.ylabel('F Score', fontsize=16)
    plt.show()


    x_axle_a=[(a+1)*interval for a in range(len(over_all_f_score))]
    plt.figure()
    plt.plot(x_axle_a, over_all_f_score,linewidth=2)
    plt.title('Overall F score on dev set of '+folder_name, fontsize=20)
    #plt.ylim(0.73,0.75)
    plt.xlabel('Epoch Time', fontsize=16)
    plt.ylabel('F Score', fontsize=16)
    plt.show()
    print(over_all_f_score)

def load_pickle(pickle_file_path,file_name):


    with open(pickle_file_path+file_name, 'rb') as f:
        train_f_score = pickle.load(f,encoding="latin1")

    return train_f_score


if __name__ == '__main__':

    cv=10

    interval=1

    for folder_name in ['baseline']: #,'CP','CP_TW','filtered'
        print(folder_name)
        dev_f_score=[]
        pickle_file_path="./data/combined/"+folder_name+"/"
        for sub_file in ['fold12','fold34','fold56','fold78','fold910']:
            dev_f_score.append(load_pickle(pickle_file_path,'pcnn_model_dev1_'+sub_file+'.pickle'))
            dev_f_score.append(load_pickle(pickle_file_path,'pcnn_model_dev2_'+sub_file+'.pickle'))
        show_train_figure(folder_name,dev_f_score,interval)





