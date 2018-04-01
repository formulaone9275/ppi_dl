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
import sys
import pickle,os

U2A_MAPPING = rewriteu2a.load_mapping()


POS = ['NN', 'NNS', 'NNP', 'NNPS', 'NNZ', 'NFP', 'VB', 'VBG', 'VBD', 'VBN',
       'VBP', 'VBZ', 'TO', 'IN', 'CC', 'RB', 'RBS', 'RBR', 'JJ', 'JJS', 'JJR',
       'DT', 'PDT', 'MD', 'MDN', 'WP', 'WP$', 'WRB', 'WDT', 'EX', 'PRP', 'PRP$',
       'POS', 'RP', 'FW', 'HYPH', 'SYM', 'CD']

def map_to_index(token):
    try:
        return VOCAB[token]
    except KeyError:
        # 0 is used for zero padding.
        return 1


DEP_RELATION_VOCAB = {}
with codecs.open('dep_relation_count.txt', 'r', encoding='utf8') as f:
    for line in f:
        relation, count = line.strip().split()
        count = int(count)
        if count < 5:
            break
        DEP_RELATION_VOCAB[relation] = len(DEP_RELATION_VOCAB) + 2
DEP_RELATION_VOCAB_SIZE = len(DEP_RELATION_VOCAB) + 2

# 0-122
def get_dep_relation_rep(relations, padding=None):
    res = []
    for relation in relations:
        index = DEP_RELATION_VOCAB.get(relation, 1)
        if padding is None:
            return index
        else:
            res.append(index)
    res = sorted(set(res))
    if len(res) < padding:
        res += [0] * (padding - len(res))
    if len(res) > padding:
        print(res)
        sys.exit()
    return res

# 0-4
def encode_entity_type(ent_type, other_encoding):
    if ent_type == 'O':
        return 1
    elif ent_type == 'PROT':
        if other_encoding == 'entity':
            return 2
        elif other_encoding == 'normal':
            return 1
    elif ent_type == 'PROT1':
        return 3
    elif ent_type == 'PROT2':
        return 4
    elif ent_type == 'PROT12':
        # Both on the same token.
        return 5
    else:
        raise ValueError

def read_sentences(filename):
    sentences, head_sents, dep_sents, sent_entity, labels = [], [], [], [], []
    count = 0

    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            if not line.startswith(('Positive', 'Negative')):
                continue

            count += 1

            line = line.strip()
            #info, tagged = line.split('\t')
            eles = line.split(' ')
            label = eles[0]
            sent = []
            head = []

            tokens = [e for e in eles if e.startswith('token:')]
            for token in tokens:
                token = token[6:]
                # Sometimes weird word contains the vertical bar.
                word, pos, ent_type, to_e1, to_e2, dep_label, dep_head = token.split('|')
                to_e1 = int(to_e1)
                to_e2 = int(to_e2)

                # could be -1.
                dep_head = dep_head
                sent.append((word, pos, ent_type, to_e1, to_e2, dep_label))


                head.append(dep_head)

            # Get label.
            head_sent = []
            for hd in head:
                if hd.strip() == 'None':
                    continue

                # could be -1, meaning no head.
                hd = int(hd)

                if hd > -1:
                    token = sent[hd]
                else:
                    token = [0] * 6
                head_sent.append(token)
            #I add the [1,1] here just to make the shape right for tensorflow
            sent_entity.append([sent[-1], sent[-1]])
            dep_sents.append(sent)
            sentences.append(sent)
            head_sents.append(head_sent)
            labels.append([0, 1] if label.startswith('Positive') else [1, 0])

    return sentences, head_sents, dep_sents, sent_entity, np.array(labels)


def sentence_matrix(sentences,
                    distance_limit=20,
                    mask_p1p2=tf.flags.FLAGS.mask_p1p2,
                    mask_other=tf.flags.FLAGS.mask_other,
                    other_encoding=tf.flags.FLAGS.other_encoding):
    matrix = []
    dep_matrix = []
    token_count = 1
    missing_embed = 0
    missing_mapped = 0
    for sent in sentences:
        # Value range of attributes.
        # [word] 0: padding, 1: UNK, 2: ARG_PROTEIN, 3-end: normal tokens
        # [pos] 0: padding, 1: UNK, 2-end: normal pos
        # [ent_type] 0: padding, 1-end: normal types
        # [to_e1] 0:
        # [to_e2]
        # [to_e1_dep] 0: padding or no dep, 1-end: normal dep distance
        # [to_e2_dep] 0: padding or no dep, 1-end: normal dep distance
        # [all_dep_labels] 0: padding or no dep, 1-end: normal dep labels
        # [on_dep]: 0: not on dep, 1: on dep

        # Yifan multi-channel CNN feature.
        # [dep_label] 0: padding or no dep, 1-end: normal dep label
        # [dep_head] not use as a feature, used to construct head channel
        #print("Sentence:",sent)
        sent_mx = []
        dep_mx = []
        for (word, pos, ent_type, to_e1, to_e2, dep_label) in sent:

            token_count += 1
            if word == 0:
                # Padding token used in head seq.
                sent_mx.append([0]*6)
                continue

            if mask_p1p2 and (ent_type in ['PROT1', 'PROT2','PROT12']):
                word = '_ARG_ENTITY_'
            if mask_other and ent_type == 'PROT':
                word = '_ENTITY_'

            index = map_to_index(word)

            if tf.flags.FLAGS.vocab_size > 0:
                if index > tf.flags.FLAGS.vocab_size:
                    index = 0

            if index == 1:
                missing_embed += 1
                word = [rewriteu2a.mapchar(c, U2A_MAPPING) for c in word]
                word = ''.join(word)
                index = map_to_index(word)
                if index == 1:
                    missing_mapped += 1

            # RANGE: 0-39
            encoded_pos = 1

            if pos in POS:
                encoded_pos = POS.index(pos) + 2

            encoded_ent = encode_entity_type(ent_type, other_encoding)

            # RANGE: 0-204
            # 0: padding
            # 1-50: originally 0-49
            # 51: originally > 49
            # 52-102: originally < 0
            encoded_to_e1 = abs(to_e1) + 1 if abs(to_e1) < distance_limit else distance_limit + 1
            if to_e1 < 0:
                encoded_to_e1 += distance_limit + 1

            # 103-152: originally 0-49
            # 153: originally > 49
            # 154-204: originally < 0
            encoded_to_e2 = abs(to_e2) + 1 if abs(to_e2) < distance_limit else distance_limit + 1
            if to_e2 < 0:
                encoded_to_e2 += distance_limit + 1
            encoded_to_e2 += (distance_limit + 1) * 2

            # RANGE: 0-122
            if dep_label.strip() == 'None':
                encoded_dep = 0
            else:
                encoded_dep = get_dep_relation_rep([dep_label])

            sent_mx.append([index, encoded_ent, encoded_pos,
                            encoded_to_e1, encoded_to_e2,
                            encoded_dep])

            '''
            print(word, pos, ent_type, to_e1, to_e2, dep_label)
            print(index, encoded_pos, encoded_ent, encoded_to_e1, encoded_to_e2, encoded_dep)
            '''

        matrix.append(sent_mx)
    '''
    print(
        'sentences: {}, tokens: {}, missing embedding: {} {}, missing mapped: {} {}'.format(
            len(sentences), token_count,
            missing_embed, float(missing_embed) / token_count,
            missing_mapped, float(missing_mapped) / token_count))
    '''
    return matrix

def context_matrix(sentences):
    left, middle, right = [], [], []
    for sent in sentences:
        sent_left, sent_middle, sent_right = [], [], []
        for token in sent:
            to_e1 = token[3]
            to_e2 = token[4]
            if to_e1 <= 0 and to_e2<=0:
                sent_left.append(token)
            if (to_e1 > 0 and to_e2 < 0) or (to_e1 < 0 and to_e2 > 0):
                sent_middle.append(token)
            if 0 <= to_e2 and 0 <= to_e1:
                sent_right.append(token)

        # Put entities at the first. This is necessary as it could be
        # more than 20 words on the left. When remove extra words, we don't
        # remove the candidate entity by putting it at the beginning.
        sent_left.reverse()
        left.append(sent_left)
        middle.append(sent_middle)
        right.append(sent_right)

    matrix_left = sentence_matrix(left)
    matrix_middle = sentence_matrix(middle)
    matrix_right = sentence_matrix(right)

    return matrix_left, matrix_middle, matrix_right


def load_context_matrix(filename):
    sentences, head_sents, dep_sents, sent_entity, labels = read_sentences(filename)
    left_mx, middle_mx, right_mx = context_matrix(sentences)
    dep_mx = sentence_matrix(dep_sents, 10)
    entity_mx = sentence_matrix(sent_entity)

    return [[left_mx, 20, [0] * 6],
            [middle_mx, 80, [0] * 6],
            [right_mx, 20, [0] * 6],
            [dep_mx, 20, [0] * 6],
            ], entity_mx, labels

def load_context_matrix_v1(filename):
    sentences, head_sents, dep_sents, sent_entity, labels = read_sentences(filename)
    left_mx, middle_mx, right_mx = context_matrix(sentences)
    dep_mx = sentence_matrix(dep_sents, 10)
    entity_mx = sentence_matrix(sent_entity)

    return left_mx,middle_mx,right_mx,dep_mx,entity_mx, labels

def pad_and_prune_seq(seq, max_len, padding):
    seq_len = max_len if len(seq) > max_len else len(seq)
    seq = seq + [padding] * (max_len - len(seq))
    seq = seq[:max_len]
    return seq, seq_len

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def build_cont_dataset(filename, target):
    data, entity_mx, labels = load_context_matrix(filename)
    left_data, middle_data, right_data, dep_data = data

    left_mx, max_left_len, left_padding = left_data
    middle_mx, max_middle_len, middle_padding = middle_data
    right_mx, max_right_len, right_padding = right_data
    dep_mx, max_dep_len, dep_padding = dep_data

    writer = tf.python_io.TFRecordWriter(target)
    for left, middle, right, dep, entity, label in zip(left_mx, middle_mx, right_mx, dep_mx, entity_mx, labels):
        left, left_len = pad_and_prune_seq(left, max_left_len, left_padding)
        middle, middle_len = pad_and_prune_seq(middle, max_middle_len, middle_padding)
        right, right_len = pad_and_prune_seq(right, max_right_len, right_padding)
        dep, dep_len = pad_and_prune_seq(dep, max_dep_len, dep_padding)

        all_seq = np.concatenate((left, middle, right, dep), axis=0)

        example = tf.train.Example(features=tf.train.Features(feature={
            'seq': _float_feature(np.ravel(all_seq)),
            'entity': _float_feature(np.ravel(entity)),
            'label': _float_feature(label),
        }))

        writer.write(example.SerializeToString())
    writer.close()

def create_tensor(data, max_len, padding):
    new_len = []
    new_instances = []
    for instance in data:
        new_instance, length = pad_and_prune_seq(instance, max_len, padding)
        new_instances.append(new_instance)
        new_len.append(length)
    return np.array(new_instances), new_len

def load_tagged(filename):
    tagged_sents = []
    with codecs.open(filename, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            info, tagged = line.split('\t')
            tagged_sents.append(tagged)

    return tagged_sents

def split_tfrecord_file(filename, target, num_per_split):
    pre_filename = os.path.basename(filename).split('.')[0]
    curr_slice = 0
    slice_filename = '{}/{}_{}.tfrecords'.format(target, pre_filename, curr_slice)
    writer = tf.python_io.TFRecordWriter(slice_filename)

    for count, r in enumerate(tf.python_io.tf_record_iterator(filename)):
        writer.write(r)
        if count > 0 and count % num_per_split == 0:
            writer.close()
            curr_slice += 1
            slice_filename = '{}/{}_{}.tfrecords'.format(target, pre_filename, curr_slice)
            writer = tf.python_io.TFRecordWriter(slice_filename)

    if count % num_per_split > 0:
        writer.close()

if __name__ == '__main__':
    print('test')
    '''
    for folder_name in ['baseline','CP','CP_TW','filtered']:
        for sub_folder in ['fold12','fold34','fold56','fold78','fold910']:
            build_cont_dataset('data/combined/'+folder_name+'/'+sub_folder+'/'+folder_name+'_combined_'+sub_folder+'_shuf.txt','data/combined/'+folder_name+'/'+sub_folder+'/'+folder_name+'_combined_'+sub_folder+'_shuf.tfrecords')

            split_tfrecord_file('data/combined/'+folder_name+'/'+sub_folder+'/'+folder_name+'_combined_'+sub_folder+'_shuf.tfrecords', 'data/combined/'+folder_name+'/'+sub_folder, 128)

    '''
    for sub_folder in ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10',]:
        build_cont_dataset('data/combined/model_instance/'+sub_folder+'.txt','data/combined/model_instance/'+sub_folder+'.tfrecords')






    #build_cont_dataset('data/combined/'+folder_name+'_dev.txt','data/combined/'+folder_name+'_dev.tfrecords')
            #build_cont_dataset('data/combined/'+folder_name+'_train_dev.txt','data/combined/'+folder_name+'_train_dev.tfrecords')



























