import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.layers import Input, CuDNNLSTM, Embedding, Dense, Reshape, LSTM, TimeDistributed, Dropout, Lambda, \
    Concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.constraints import unit_norm

import numpy as np

from keras_lm.language_model.tied_embeddings import TiedEmbeddingsTransposed
from keras_lm.language_model.qrnn import QRNN


def build_language_model(num_words, embedding_size=300, rnn_sizes=(1024, 512),
                         dropout=0.1, dropouth=0.3, dropouti=0.2, dropoute=0.1, wdrop=0.5,
                         tie_weights=True, use_qrnn=False, use_gpu=True):

    inp = Input(shape=(None,), name='input')
    emb = Embedding(num_words, embedding_size, name='embedding')
    emb_inp = emb(inp)
    emb_inp = Dropout(dropouti)(emb_inp)

    if use_qrnn:
        rnn = QRNN(rnn_sizes[0], return_sequences=True, window_size=2)(emb_inp)
        for rnn_size in rnn_sizes[1:]:
            rnn = QRNN(rnn_size, return_sequences=True, window_size=1)(rnn)
        rnn = QRNN(embedding_size, return_sequences=True, window_size=1, name='final_rnn_layer')(rnn)
    else:
        RnnUnit = CuDNNLSTM if use_gpu else LSTM
        rnn = RnnUnit(rnn_sizes[0], return_sequences=True)(emb_inp)
        for rnn_size in rnn_sizes[1:]:
            rnn = RnnUnit(rnn_size, return_sequences=True)(rnn)
        rnn = RnnUnit(embedding_size, return_sequences=True, name='final_rnn_layer')(rnn)

    if tie_weights:
        logits = TimeDistributed(TiedEmbeddingsTransposed(tied_to=emb, activation='softmax'))(rnn)
    else:
        logits = TimeDistributed(Dense(num_words, activation='softmax'))(rnn)
    out = Dropout(dropout)(logits)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr=3e-4, beta_1=0.8, beta_2=0.99), loss='sparse_categorical_crossentropy')
    return model


def build_many_to_one_language_model(num_words, embedding_size=300, use_qrnn=False, use_gpu=True):
    inp = Input(shape=(None,), name='input')
    emb = Embedding(num_words, embedding_size, name='embedding')
    emb_inp = emb(inp)

    if use_qrnn:
        rnn = QRNN(1024, return_sequences=True, window_size=2)(emb_inp)
        rnn = QRNN(1024, return_sequences=True, window_size=1)(rnn)
        rnn = QRNN(embedding_size, return_sequences=False,window_size=1, name='final_rnn_layer')(rnn)
    else:
        RnnUnit = CuDNNLSTM if use_gpu else LSTM
        rnn = RnnUnit(1024, return_sequences=True)(emb_inp)
        rnn = RnnUnit(1024, return_sequences=True)(rnn)
        rnn = RnnUnit(embedding_size, return_sequences=False, name='final_rnn_layer')(rnn)

    out = TiedEmbeddingsTransposed(tied_to=emb, activation='softmax')(rnn)
    model = Model(inputs=inp, outputs=out)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model



if __name__ == '__main__':
    model = build_language_model(num_words=100)
    model.summary()

    simple_model = build_many_to_one_language_model(num_words=100, embedding_size=300)
    simple_model.summary()


