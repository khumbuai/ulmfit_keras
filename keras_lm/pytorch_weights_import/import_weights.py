"""
Imports the pretrained weightsform pytorch to keras.
"""
from keras import backend as K

import torch
import pickle
from collections import defaultdict
import numpy as np

from keras_lm.language_model.model import build_language_model
from keras_lm.language_model.train import evaluate_model

# The pretrained weights and the itos_wt103.pkl file can be found on http://files.fast.ai/models/wt103/
PYTORCH_WEIGTHS_FILEPATH = 'assets/weights/fwd_wt103.h5'
PYTORCH_IDX2WORD_FILEPATH = 'assets/wikitext-103/itos_wt103.pkl'
UNKNOWN_TOKEN = '<unk>'

# Pytorch model_parameters
em_sz,nh,nl = 400, 1150, 3

# Load the weights of the pretrained pytorch model
wgts = torch.load(PYTORCH_WEIGTHS_FILEPATH, map_location=lambda storage, loc: storage)

# for key, value in wgts.items():
#     print(key)
#     print(value.numpy().shape)
    # 0.encoder.weight
    # (238462, 400)

    # 0.encoder_with_dropout.embed.weight
    # (238462, 400)

    # 0.rnns.0.module.weight_ih_l0
    # (4600, 400)
    # 0.rnns.0.module.bias_ih_l0
    # (4600,)
    # 0.rnns.0.module.bias_hh_l0
    # (4600,)
    # 0.rnns.0.module.weight_hh_l0_raw
    # (4600, 1150)

    # 0.rnns.1.module.weight_ih_l0
    # (4600, 1150)
    # 0.rnns.1.module.bias_ih_l0
    # (4600,)
    # 0.rnns.1.module.bias_hh_l0
    # (4600,)
    # 0.rnns.1.module.weight_hh_l0_raw
    # (4600, 1150)

    # 0.rnns.2.module.weight_ih_l0
    # (1600, 1150)
    # 0.rnns.2.module.bias_ih_l0
    # (1600,)
    # 0.rnns.2.module.bias_hh_l0
    # (1600,)
    # 0.rnns.2.module.weight_hh_l0_raw
    # (1600, 400)

    # 1.decoder.weight
    # (238462, 400)


pytorch_idx2word = itos = pickle.load(open(PYTORCH_IDX2WORD_FILEPATH, 'rb'))

corpus = pickle.load(open('assets/wikitext-103/wikitext-103.corpus','rb'))
word2idx = defaultdict(lambda: word2idx[UNKNOWN_TOKEN], corpus.word2idx)

def pytorch_int2keras_int(pytorch_int):
    """
    Maps itos (from pytorch saved model) to word2idx (as used in Keras implementation).
    :param pytorch_int:
    :return:
    """
    return word2idx[pytorch_idx2word[pytorch_int]]


def create_embedding_weights():
    embedding_weights =  wgts['0.encoder.weight'].numpy()
    num_words = embedding_weights.shape[0]

    changed_numbering = [pytorch_int2keras_int(i) for i in range(num_words)]
    embedding_weights = embedding_weights[changed_numbering]

    return embedding_weights.reshape((1, -1, em_sz))


def create_rnn_weights(i):
    """
    Trainable layers in the tensorflow LSTM layer: (example first_rnn_layer)
    [<tf.Variable 'lstm_1/kernel:0' shape=(400, 4600) dtype=float32_ref>,
     <tf.Variable 'lstm_1/recurrent_kernel:0' shape=(1150, 4600) dtype=float32_ref>,
    <tf.Variable 'lstm_1/bias:0' shape=(4600,) dtype=float32_ref>]
    :param int i: number of rnn layer
    :return:
    """
    prefix = '0.rnns.' + str(i) + '.module.'

    ih_weight = wgts[prefix + 'weight_ih_l0'].numpy().T  # shape (input_dim, 4 * rnn_size)
    ih_bias = wgts[prefix + 'bias_ih_l0'].numpy()  # shape (4 * rnn_size,)

    hh_weights= wgts[prefix + 'weight_hh_l0_raw'].numpy().T  # shape (rnn_size, 4 * rnn_size)
    hh_bias = wgts[prefix + 'bias_hh_l0'].numpy()  # (4 * rnn_size,)

    # pytorch uses two biases, whereas tensorflow only has one shared bias layer
    bias = 0.5 * (ih_bias + hh_bias)

    return ih_weight, hh_weights, bias


if __name__ == '__main__':

    # 1. Grap weights from Pytorch model
    embedding_weights = create_embedding_weights()

    rnn_weights = [create_rnn_weights(i) for i in range(3)]

    # 2. Initialize keras model with pretrained weights
    language_model = build_language_model(num_words=embedding_weights.shape[1],
                                           embedding_size=em_sz,
                                           rnn_sizes=(1150, 1150, 1150),
                                           tie_weights=True,
                                           use_qrnn=False,
                                           use_gpu=False,
                                           only_last=False)
    language_model.summary()

    embedding_layer = language_model.get_layer('embedding')
    embedding_layer.set_weights(embedding_weights)

    first_rnn_layer = language_model.layers[3]
    second_rnn_layer = language_model.layers[4]
    third_rnn_layer = language_model.layers[5]

    for i, rnn_layer in enumerate([first_rnn_layer, second_rnn_layer, third_rnn_layer]):
        rnn_layer.set_weights(rnn_weights[i])

    # 3. Save model
    language_model.save('assets/language_model.hdf5', overwrite=True)

    # 4. Evaluate language model
    evaluate_model(language_model, word2idx, 'i feel sick and go to the next', num_predictions=20)

