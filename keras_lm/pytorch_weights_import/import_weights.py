"""
Imports the pretrained weights from pytorch to keras.
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

# Load the weights of the pretrained pytorch model
wgts = torch.load(PYTORCH_WEIGTHS_FILEPATH, map_location=lambda storage, loc: storage)

"""
for key, value in wgts.items():
    print(key)
    print(value.numpy().shape)
    
0.encoder.weight
(238462, 400)

0.encoder_with_dropout.embed.weight
(238462, 400)

0.rnns.0.module.weight_ih_l0
(4600, 400)
0.rnns.0.module.bias_ih_l0
(4600,)
0.rnns.0.module.bias_hh_l0
(4600,)
0.rnns.0.module.weight_hh_l0_raw
(4600, 1150)

0.rnns.1.module.weight_ih_l0
(4600, 1150)
0.rnns.1.module.bias_ih_l0
(4600,)
0.rnns.1.module.bias_hh_l0
(4600,)
0.rnns.1.module.weight_hh_l0_raw
(4600, 1150)

0.rnns.2.module.weight_ih_l0
(1600, 1150)
0.rnns.2.module.bias_ih_l0
(1600,)
0.rnns.2.module.bias_hh_l0
(1600,)
0.rnns.2.module.weight_hh_l0_raw
(1600, 400)

1.decoder.weight
(238462, 400)
"""

def create_embedding_weights():
    embedding_weights = wgts['0.encoder.weight'].numpy()

    return embedding_weights.reshape((1, -1, 400))


def create_rnn_weights(i, use_gpu=True):
    """
    Fetches the weights form the pytorch LSTM layer and transforms it to weights which can be used
    for tensorflow LSTM layers
    :param int i: number of rnn layer
    :return:
    """

    """
    We need to ensure that the weight matrices map correctly from pytorch to keras.
    Both implementations use the same ordering in the CuDNNLSTM implementation.
    
    Trainable layers in the tensorflow LSTM layer: (example first_rnn_layer)
    [<tf.Variable 'lstm_1/kernel:0' shape=(400, 4600) dtype=float32_ref>,
     <tf.Variable 'lstm_1/recurrent_kernel:0' shape=(1150, 4600) dtype=float32_ref>,
    <tf.Variable 'lstm_1/bias:0' shape=(4600,) dtype=float32_ref>]
    
    CuDNNLSTM implementation in keras
    https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L324

    self.kernel_i = self.kernel[:, :self.units]
    self.kernel_f = self.kernel[:, self.units: self.units * 2]
    self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
    self.kernel_o = self.kernel[:, self.units * 3:]

    self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
    self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
    self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
    self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

    self.bias_i_i = self.bias[:self.units]
    self.bias_f_i = self.bias[self.units: self.units * 2]
    self.bias_c_i = self.bias[self.units * 2: self.units * 3]
    self.bias_o_i = self.bias[self.units * 3: self.units * 4]

    self.bias_i = self.bias[self.units * 4: self.units * 5]
    self.bias_f = self.bias[self.units * 5: self.units * 6]
    self.bias_c = self.bias[self.units * 6: self.units * 7]
    self.bias_o = self.bias[self.units * 7:]
    
    CuDNNLSTM implementation in torch
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
    weight_ih_l0 - (W_ii, W_if, W_ig, W_io)
    weight_hh_l0 - (W_hi, W_hf, W_hg, W_ho)
    bias_ih_l0 - (b_ii, b_if, b_ig, b_io)
    bias_hh_l0 - (b_hi, b_hf, b_hg, b_ho)
    """

    prefix = '0.rnns.' + str(i) + '.module.'

    ih_weight = wgts[prefix + 'weight_ih_l0'].numpy().T  # shape (input_dim, 4 * rnn_size)
    ih_bias = wgts[prefix + 'bias_ih_l0'].numpy()  # shape (4 * rnn_size,)

    hh_weights= wgts[prefix + 'weight_hh_l0_raw'].numpy().T  # shape (rnn_size, 4 * rnn_size)
    hh_bias = wgts[prefix + 'bias_hh_l0'].numpy()  # (4 * rnn_size,)

    #  tensorflow has one shared bias layer for cpu, but two for gpu (maybe due to parallel processing)
    if not use_gpu:
        bias = 0.5 * (ih_bias + hh_bias)
    else:
        bias = np.concatenate([ih_bias, hh_bias])

    return ih_weight, hh_weights, bias


if __name__ == '__main__':
    LANGUAGE_MODEL_PARAMS = {'embedding_size': 400, 'rnn_sizes': (1150, 1150), 'use_gpu': False,
                             'dropout': 0.1, 'tie_weights': True, 'use_qrnn': False, 'only_last': False
                             }

    # 1. Grap weights from Pytorch model
    embedding_weights = create_embedding_weights()

    rnn_weights = [create_rnn_weights(i, use_gpu=LANGUAGE_MODEL_PARAMS['use_gpu']) for i in range(3)]

    # 2. Initialize keras model with pretrained weights
    language_model = build_language_model(num_words=embedding_weights.shape[1],
                                          **LANGUAGE_MODEL_PARAMS)
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
    with open(PYTORCH_IDX2WORD_FILEPATH, 'rb') as f:
        idx2word = pickle.load(f)

    word2idx = {word: idx for idx, word in enumerate(idx2word)}

    evaluate_model(language_model, word2idx, 'i feel sick and go to the next', num_predictions=20)
