import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.layers import Input, CuDNNLSTM, Embedding, Dense, Reshape, LSTM, TimeDistributed, Dropout, Lambda, \
    Concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.constraints import unit_norm

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


def build_fast_language_model(num_words, embedding_size=300, dropouti=0.2, rnn_sizes=(1024, 512),
                              use_qrnn=False, use_gpu=True):
    """
    Adatped from http://adventuresinmachinelearning.com/word2vec-keras-tutorial/
    :param num_words:
    :param embedding_size:
    :param use_gpu:
    :return:
    """
    # create some input variables
    inp = Input(shape=(None,), name='input')
    inp_target = Input(shape=(None,), name='target')  # this is the shifted sequence

    emb = Embedding(num_words, embedding_size, name='embedding',  embeddings_constraint=unit_norm(axis=1))

    emb_inp = emb(inp)
    emb_inp = Dropout(dropouti)(emb_inp)

    emb_target = emb(inp_target)

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

    def scalar_product(vector1, vector2):
        #TODO: find a cleaner way to perform the timedistributed dot product
        helper_tensor = Concatenate()([vector1, vector2])
        reshaped = Reshape((-1, embedding_size, 2))(helper_tensor)

        def tensor_product(x):
            a = x[:, :, :, 0]
            b = x[:, :, :, 1]
            y = K.sum(a * b, axis=-1, keepdims=False)
            return y
        return Lambda(tensor_product)(reshaped)  # similarity is between -1 and 1

    similarity = scalar_product(rnn, emb_target)
    # dissimilarity implements Skip-Gram negative sampling.
    # The current word embedding should be orthogonal to the word embedding of the following word.
    dissimilarity = scalar_product(emb_inp, emb_target)

    model = Model(inputs=[inp, inp_target], outputs=[similarity, dissimilarity])
    model.compile(loss='mse', optimizer=Adam(lr=3e-4, beta_1=0.8, beta_2=0.99))
    return model


def fast_language_model_evaluation(model):
    """
    Builds a new model on top of the fast_language_model which can be used for evaluation of the predictions.
    :param model: trained model instance of the model defined in build_fast_language_model
    :return:
    """
    inp = [layer.input for layer in model.layers if layer.name == 'input'][0]
    rnn_output = [layer.output for layer in model.layers if layer.name == 'final_rnn_layer'][0]
    embedding_layer = [layer for layer in model.layers if layer.name == 'embedding'][0]

    # no activation since we use argmax for prediction, probably faster.
    out = TiedEmbeddingsTransposed(tied_to=embedding_layer, activation=None)(rnn_output)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=3e-4, beta_1=0.8, beta_2=0.99))
    return model


if __name__ == '__main__':
    model = build_language_model(num_words=100)
    model.summary()

    simple_model = build_many_to_one_language_model(num_words=100, embedding_size=300)
    simple_model.summary()

    fast_model = build_fast_language_model(num_words=100, embedding_size=300, use_gpu=False)
    fast_model.summary()

    fast_evaluation = fast_language_model_evaluation(fast_model)
    fast_evaluation.summary()
