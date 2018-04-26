from __future__ import print_function
import codecs
from word_embedding import VOCAB,EMBEDDING
import tensorflow as tf
import numpy as np
from nxml2txt.src import rewriteu2a
from collections import OrderedDict, Counter
import sklearn as sk
from glob import glob
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
#import matplotlib.pyplot as plt
from flags import *
import random

U2A_MAPPING = rewriteu2a.load_mapping()



DISTANCE_ENCODING = []
for i in range(10):
    DISTANCE_ENCODING.append([0] * (9-i) + [1] * i)

DISTANCE_MAPPING = OrderedDict([
    (0, DISTANCE_ENCODING[0]),
    (1, DISTANCE_ENCODING[1]),
    (2, DISTANCE_ENCODING[2]),
    (3, DISTANCE_ENCODING[3]),
    (4, DISTANCE_ENCODING[4]),
    (5, DISTANCE_ENCODING[5]),
    (6, DISTANCE_ENCODING[6]),
    (11, DISTANCE_ENCODING[7]),
    (21, DISTANCE_ENCODING[8]),
    (31, DISTANCE_ENCODING[9]),
])


def encode_distance(distance):
    sign = 1 if distance > 0 else 0
    distance = abs(distance)

    if 0 <= distance <= 5:
        return [sign] + DISTANCE_MAPPING[distance]
    elif 6 <= distance <= 10:
        return [sign] + DISTANCE_MAPPING[6]
    elif 11 <= distance <= 20:
        return [sign] + DISTANCE_MAPPING[11]
    elif 21 <= distance <= 30:
        return [sign] + DISTANCE_MAPPING[21]
    elif 31 <= distance:
        return [sign] + DISTANCE_MAPPING[31]


POS_ENCODING = []
for i in range(8):
    encoding = [0] * 8
    encoding[i] = 1
    POS_ENCODING.append(encoding)

POS_MAPPING = {
    'NN': POS_ENCODING[0],
    'NNS': POS_ENCODING[0],
    'NNP': POS_ENCODING[0],
    'NNPS': POS_ENCODING[0],
    'NNZ': POS_ENCODING[0],
    'NFP': POS_ENCODING[0],

    'VB': POS_ENCODING[1],
    'VBG': POS_ENCODING[1],
    'VBD': POS_ENCODING[1],
    'VBN': POS_ENCODING[1],
    'VBP': POS_ENCODING[1],
    'VBZ': POS_ENCODING[1],

    'TO': POS_ENCODING[2],
    'IN': POS_ENCODING[2],
    'CC': POS_ENCODING[2],

    'RB': POS_ENCODING[3],
    'RBS': POS_ENCODING[3],
    'RBR': POS_ENCODING[3],
    'JJ': POS_ENCODING[3],
    'JJS': POS_ENCODING[3],
    'JJR': POS_ENCODING[3],
    'DT': POS_ENCODING[3],
    'PDT': POS_ENCODING[3],
    'MD': POS_ENCODING[3],
    'MDN': POS_ENCODING[3],

    'WP': POS_ENCODING[4],
    'WP$': POS_ENCODING[4],
    'WRB': POS_ENCODING[4],
    'WDT': POS_ENCODING[4],
    'EX': POS_ENCODING[4],

    'PRP': POS_ENCODING[5],
    'PRP$': POS_ENCODING[5],

    'POS': POS_ENCODING[6],
    'RP': POS_ENCODING[6],
    'FW': POS_ENCODING[6],
    'HYPH': POS_ENCODING[6],
    'SYM': POS_ENCODING[6],

    'CD': POS_ENCODING[7],
}


def encode_pos(pos_tag):
    if pos_tag in POS_MAPPING:
        return POS_MAPPING[pos_tag]
    else:
        return [0]*8


def map_to_index(token):
    try:
        return VOCAB[token]
    except KeyError:
        return 1

def encode_entity_type(ent_type, other_encoding):
    if ent_type == 'O':
        return [1, 0, 0, 0]
    elif ent_type == 'PROT':
        if other_encoding == 'entity':
            return [0, 1, 0, 0]
        elif other_encoding == 'normal':
            return [1, 0, 0, 0]
    elif ent_type == 'PROT1':
        return [0, 0, 1, 0]
    elif ent_type == 'PROT2':
        return [0, 0, 0, 1]
    elif ent_type == 'PROT12':
        return [0, 0, 1, 1]

DEP_RELATION_VOCAB = {}
with codecs.open('dep_relation_count.txt', 'r', encoding='utf8') as f:
    for line in f:
        relation, count = line.strip().split()
        count = int(count)
        if count < 5:
            break
        DEP_RELATION_VOCAB[relation] = len(DEP_RELATION_VOCAB) + 1
DEP_RELATION_VOCAB_SIZE = len(DEP_RELATION_VOCAB)+1


def get_dep_relation_rep(relation):
    vec = [0] * DEP_RELATION_VOCAB_SIZE

    index = DEP_RELATION_VOCAB.get(relation, 0)
    #print(index)
    vec[index] = 1
    return vec


def read_sentences(filename):
    sentences, head_sents, dep_sents, labels = [], [], [], []
    count = 0

    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            count += 1

            line = line.strip()
            info = line.split(' ')
            label = info[0]
            sent = []
            head = []

            tokens = [e for e in info if e.startswith('token:')]
            for token in tokens:
                token = token[6:]
                word, pos, ent_type, to_e1, to_e2, dep_labels, dep_heads = token.split('|')
                to_e1 = int(to_e1)
                to_e2 = int(to_e2)

                sent.append((word, pos, ent_type, to_e1, to_e2, dep_labels))
                head.append(dep_heads)
            # Get head word
            #print(head)
            head_sent = []
            for hd in head:
                #print(hd.strip())
                if hd.strip() == 'None':
                    continue
                else:
                    hd = int(hd)
                    token = sent[hd]
                    head_sent.append(token)

            sentences.append(sent)
            head_sents.append(head_sent)

            labels.append([0, 1] if label.startswith('Positive') else [1, 0])

    return sentences, head_sents, np.array(labels)


def sentence_matrix(sentences,
                    mask_p1p2=tf.flags.FLAGS.mask_p1p2,
                    mask_other=tf.flags.FLAGS.mask_other,
                    other_encoding=tf.flags.FLAGS.other_encoding):
    matrix = []
    token_count = 0
    missing_embed = 0
    missing_mapped = 0
    for sent in sentences:
        sent_mx = []
        for word, pos, ent_type, to_e1, to_e2, dep_labels in sent:
            token_count += 1
            if mask_p1p2 and ent_type == 'PROT1':
                #print(word)
                word = '_PROTEIN_'
            if mask_p1p2 and ent_type == 'PROT2':
                #print(word)
                word = '_PROTEIN_'
            if mask_p1p2 and ent_type == 'PROT12':
                #print(word)
                word = '_PROTEIN_'
            if mask_other and ent_type == 'PROT':
                word = '_ENTITY_'
            index = map_to_index(word)

            if index == 1:
                missing_embed += 1
                word = [rewriteu2a.mapchar(c, U2A_MAPPING) for c in word]
                word = ''.join(word)
                index = map_to_index(word)
                if index == 1:
                    missing_mapped += 1
            encoded_pos = encode_pos(pos)
            encoded_ent = encode_entity_type(ent_type, other_encoding)
            encoded_to_e1 = encode_distance(to_e1)
            encoded_to_e2 = encode_distance(to_e2)
            encoded_dep = get_dep_relation_rep(dep_labels)
            '''
            print(index)
            print(encoded_ent)
            print(encoded_pos)
            print(encoded_to_e1)
            print(encoded_to_e2)
            print(encoded_dep)
            '''
            sent_mx.append([index] + encoded_ent + encoded_pos +
                           encoded_to_e1+encoded_to_e2+encoded_dep)
            #print(len(sent_mx[0]))
            '''
            print(word, pos, ent_type, to_e1, to_e2, dep_labels,
                  index, encoded_pos, encoded_ent, encoded_to_e1, encoded_to_e2, encoded_dep)
            '''
        '''
        if len(sent_mx) < max_length:
            sent_mx += [[0] * (33+DEP_RELATION_VOCAB_SIZE)]*(max_length-len(sent_mx))
        '''

        matrix.append(sent_mx)

    print('sentences: {}, tokens: {}, missing embedding: {} {}, missing mapped: {} {}'.format(
        len(sentences), token_count,
        missing_embed, float(missing_embed)/token_count,
        missing_mapped, float(missing_mapped)/token_count))
    return matrix

def load_sentence_matrix(filename):

    sentences, head_sents,  labels = read_sentences(filename)
    sent_mx = sentence_matrix(sentences)
    #print(np.shape(sent_mx))
    #print(sent_mx[0])
    head_mx = sentence_matrix(head_sents)

    return [[sent_mx, 160, [0]*155],
            [head_mx, 160, [0]*155]], labels

def pad_and_prune_seq(seq, max_len, padding):
    seq_len = max_len if len(seq) > max_len else len(seq)
    seq = seq + [padding] * (max_len - len(seq))
    seq = seq[:max_len]

    return seq, seq_len

def build_dataset(filename, target):
    data, labels = load_sentence_matrix(filename)
    #print(target)
    sent_data, head_data = data
    #(sent_data)
    sent_mx, max_sent_len, sent_padding = sent_data
    head_mx, max_head_len, head_padding = head_data
    #dep_mx, max_dep_len, dep_padding = dep_data
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


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_tfrecord_data(filename,target_filename):
    input_data,label=build_dataset(filename, target_filename)
    with codecs.open('./data/model_instance/results.txt','w+', encoding='utf8') as f:
        for ii in range(len(input_data[0])):
            for jj in range(len(input_data[0][ii])):
                f.write(str(input_data[0][ii][jj]))
                f.write(' ')
            f.write('\n')
    f.close()
    #print(label)
    #concantenate the embedding vector
    input_data_all=[]
    label_list=[]
    writer = tf.python_io.TFRecordWriter(target_filename)
    for ii in range(len(input_data)):
        input_data_all_temp=[]
        #print(len(input_data))
        for jj in range(len(input_data[0])):
            temp=np.concatenate((EMBEDDING[input_data[ii][jj][0]],input_data[ii][jj][1:]))
            input_data_all_temp.append(temp)
            #print(temp)
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

def random_and_divide_file_v0(filename,output_folder,number):
    file_dict={}
    count_line=0

    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            file_dict[count_line] = line.strip()

            count_line+=1
    index_list=list(range(len(file_dict)))
    random.shuffle(index_list)


    for i in range(number):
        file_object=codecs.open(output_folder+"fold"+str(i+1)+".txt",'w+',encoding='utf8')
        for j in range(int(len(file_dict)/number)*i,int(len(file_dict)/number)*(i+1)):

            file_object.write(file_dict[index_list[j]])
            file_object.write('\n')
        file_object.close()

def random_and_divide_file(filename,output_folder,number):
    file_dict={}
    count_line=0
    document_id=[]
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            file_dict[count_line] = line.strip()
            line_spilt=line.strip().split(' ')
            if line_spilt[1] not in document_id:
                document_id.append(line_spilt[1])
            #print(line_spilt[1])
            count_line+=1
    index_list=list(range(len(file_dict)))
    random.shuffle(index_list)
    random.shuffle(document_id)
    #docu_number=int(len(document_id)/number)
    #divide the file based on the document id
    divided_document={}
    #initilize the dict
    for ii in range(number):
        divided_document[ii]=[]
    for jj in range(len(document_id)):
        docu_index=jj%number
        divided_document[docu_index].append(document_id[jj])

    for ii in range(number):
        print(divided_document[ii])

    for i in range(number):
        file_object=codecs.open(output_folder+"fold"+str(i+1)+".txt",'w+',encoding='utf8')
        for j in range(len(file_dict)):
            #print(index_list[j],'-',len(file_dict))
            info=file_dict[index_list[j]].split(' ')
            #print(info)
            if info[1] in divided_document[i]:
                file_object.write(file_dict[index_list[j]])
                file_object.write('\n')
        file_object.close()

def count_neg_pos_instance(filename):
    count_dict={}
    document_id=[]
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            line_spilt=line.strip().split(' ')
            if line_spilt[1] not in document_id:
                document_id.append(line_spilt[1])
            if line_spilt[1] not in count_dict.keys() and line_spilt[0].startswith('Positive'):
                count_dict[line_spilt[1]]=[1,0]
            elif line_spilt[1] not in count_dict.keys() and line_spilt[0].startswith('Negative'):
                count_dict[line_spilt[1]]=[0,1]
            elif line_spilt[0].startswith('Positive'):
                count_dict[line_spilt[1]][0]+=1
            elif line_spilt[0].startswith('Negative'):
                count_dict[line_spilt[1]][1]+=1


    for ii in range(len(document_id)):
        print(document_id[ii],count_dict[document_id[ii]])

def randomize_file(filename,output_folder):
    file_dict={}
    count_line=0
    document_id=[]
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            file_dict[count_line] = line.strip()
            count_line+=1
    index_list=list(range(len(file_dict)))
    random.shuffle(index_list)
    #write into the new file
    file_object=codecs.open(output_folder+"Randomized.txt",'w+',encoding='utf8')
    for j in range(len(file_dict)):
      file_object.write(file_dict[index_list[j]])
      file_object.write('\n')
    file_object.close()


if __name__ == '__main__':
    random_and_divide_file('./data/aimed.txt','./data/model_size/',4)
    '''
    #randomize the data and divide the file
    for file_name in ['baseline','CP','CP_TW','filtered']:

        random_and_divide_file_v0('./data/pretrain/'+file_name+'_shuf.txt','./data/pretrain/'+file_name+'/',100)

        for ii in range(100):
            filename1='data/pretrain/'+file_name+'/fold'+str(ii+1)+'.txt'
            filename2='data/pretrain/'+file_name+'/aimed_pretrain'+str(ii+1)+'.tfrecords'
            print(filename1)
            build_tfrecord_data(filename1,filename2)


    #random_and_divide_file_v0('./data/aimed.txt','./data/model_size/',3)
    '''
    '''
    for ii in range(10):
        filename1='data/model_instance/fold'+str(ii+1)+'.txt'
        filename2='data/model_instance/aimed_cross_validataion'+str(ii+1)+'.tfrecords'
        print(filename1)
        build_tfrecord_data(filename1,filename2)
    '''
    #build_tfrecord_data('./data/model5/fold_test.txt','data/model5/aimed_cross_validataion_test.tfrecords')
    #build the data from distant supervision
    #randomize_file('./data/train_filtered.txt','./data/')
    #build_tfrecord_data('data/Randomized.txt','data/Randomized.tfrecords')
    #count_neg_pos_instance('./data/aimed.txt')