import data
from utils import batchify, get_batch, data_gen2
import numpy as np

import os
#from keras.utils import multi_gpu_model
from tqdm import tqdm
#import tensorflow as tf
import pickle
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.layers import Input, CuDNNLSTM, Embedding, Dense, TimeDistributed, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tied_embeddings import TiedEmbeddingsTransposed
from qrnn import QRNN
#corpus = data.Corpus('assets/wikitext-103/')
#pickle.dump(corpus,open('wikitext-103.corpus','wb'))

corpus = pickle.load(open('wikitext-103.corpus','rb'))


eval_batch_size = 10
test_batch_size = 1
batch_size = 32
SEQ_LEN = 50
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, test_batch_size)


train_gen = data_gen2(train_data,SEQ_LEN, batch_size=batch_size)
valid_gen = data_gen2(val_data,SEQ_LEN, batch_size=eval_batch_size)


num_words = len(corpus.word2idx) +1

def build_language_model(dropout=0.1, dropouth=0.3, dropouti=0.2, dropoute=0.1, wdrop=0.5, tie_weights=True, use_qrnn=False):

    inp = Input(shape=(None,))
    emb = Embedding(num_words,300)
    emb_inp = emb(inp)
    emb_inp = Dropout(dropouti)(emb_inp)

    if use_qrnn:
        rnn = QRNN(1024, return_sequences=True, window_size=2)(emb_inp)
        rnn = QRNN(1024, return_sequences=True, window_size=1)(rnn)
        rnn = QRNN(300, return_sequences=True,window_size=1)(rnn)
    else:
        rnn = CuDNNLSTM(1024, return_sequences=True)(emb_inp)
        rnn = CuDNNLSTM(1024, return_sequences=True)(rnn)
        rnn = CuDNNLSTM(300, return_sequences=True)(rnn)

    if tie_weights:
        logits = TimeDistributed(TiedEmbeddingsTransposed(tied_to=emb,activation='softmax'))(rnn)
    else:
        logits = TimeDistributed(Dense(num_words, activation='softmax'))(rnn)
    out = Dropout(dropout)(logits)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr=3e-4,beta_1=0.8,beta_2=0.99), loss='sparse_categorical_crossentropy')
    model.summary()
    return model

early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint('model.hdf5', save_weights_only=True)
model = build_language_model()
model.fit_generator(train_gen,
                    steps_per_epoch=train_data.shape[0]//SEQ_LEN,
                    epochs=10,
                    validation_data=valid_gen,
                    validation_steps=val_data.shape[0]//SEQ_LEN,
                    callbacks=[early_stop,check_point])



#model.evaluate_generator(valid_gen, steps=val_data.shape[0]//SEQ_LEN, verbose=True)


test_sentence ='i am sick . hence i go to the'.split()
encoded_sentence = [corpus.word2idx[w] for w in test_sentence]
for i in range(5):
    pred = model.predict(encoded_sentence)
    answer = np.argmax(pred,axis = 2)
    encoded_sentence.append([a[0] for a in answer][-1])

print(' '.join([corpus.idx2word[i[0]] for i in answer]))
print(' '.join([corpus.idx2word[i] for i in encoded_sentence]))





#output = TimeDistributed(Dense(opt.word_vocab_size, activation='softmax'))(x)