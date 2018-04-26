from gensim.models import KeyedVectors
import numpy as np



w2v_model = KeyedVectors.load_word2vec_format(
        'data2/PubMed-shuffle-win-2.bin', binary=True)

VOCAB_SIZE = len(w2v_model.vocab)+4

print("vocab size:",VOCAB_SIZE)
EMBEDDING_DIM = w2v_model.vector_size
EMBEDDING = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
EMBEDDING[1,:] = np.random.uniform(-0.001, 0.001, size=[200,])
EMBEDDING[2,:] = np.random.uniform(-0.001, 0.001, size=[200,])
EMBEDDING[3,:] = np.random.uniform(-0.001, 0.001, size=[200,])

VOCAB = {'_UNK_': 1, '_ENTITY_': 2, '_ARG_ENTITY_': 3}
for i in range(4, VOCAB_SIZE):
    word = w2v_model.index2word[i-4]
    VOCAB[word] = i

    vector = w2v_model[word]
    EMBEDDING[i] = vector
    if i==5:
        print('word:',word)


del w2v_model
