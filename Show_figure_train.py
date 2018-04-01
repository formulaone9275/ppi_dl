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
    pretrained_indicator=True

    cv_indicator=False
    interval=25
    pickle_file_pretrain_indicator ='pretrained' if pretrained_indicator else ''
    print(pickle_file_pretrain_indicator)
    for folder_name in ['baseline','CP','CP_TW','filtered']:
        print(folder_name)
        #folder_name_test="CP"
        pickle_file_path="./model/"+folder_name+"/"

        train_f_score=load_pickle(pickle_file_path,'test_fscore.pickle')
        print(train_f_score)
        print(len(train_f_score))
        show_train_figure(folder_name,train_f_score,interval)





