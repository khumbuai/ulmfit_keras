from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, BatchNormalization, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

import pickle
from collections import defaultdict

from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

from keras_lm.transfer_model.multilayer_optimizer import LRMultiplierSGD
from keras_lm.language_model.model import build_language_model


def tokenize(df, word2idx, maxlen=50):
    """Right pads the sequence"""
    word2idx = defaultdict(lambda: word2idx['<unk>'], word2idx)
    df = df.apply(lambda x: [word2idx[word.lower()] for word in x.split()])
    return pad_sequences(df['Phrase'], maxlen=maxlen)


def train_transfer_model(transfer_model, train_gen, steps_per_epoch, learning_rates):
    """
    Implementation of language model finetuning by unfreezing the layer step by step.
    :param language_model:
    :param model_description:
    :param list learning_rates: list of learning rates.
    :return:
    """
    K.clear_session()

    for layer in transfer_model.layers:
        layer.trainable = False

    for i, layer in enumerate(reversed(transfer_model.layers)):
        layer.trainable = True

        transfer_model.compile(loss=sparse_categorical_crossentropy, optimizer=LRMultiplierSGD(lr=learning_rates[i],
                                                                                               momentum=0., decay=0.,
                                                                                               nesterov=False))

        transfer_model.fit_generator(train_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=1,
                                     )

    return transfer_model


def language_classification_model(language_model, num_labels, lr=0.0001, lr_d=0.0, kernel_size1=3, kernel_size2=2,
                                  dense_units=128, dropout=0.1, filters=32):
    """
    Transfer model for language classification.
    :param language_model:
    :param lr:
    :param lr_d:
    :param kernel_size1:
    :param kernel_size2:
    :param dense_units:
    :param dropout:
    :param filters:
    :return:
    """
    for layer in language_model.layers:
        layer.trainable = False

    lstm_output = [layer.output for layer in language_model.layers if layer.name == 'final_rnn_layer'][0]

    x1 = Conv1D(filters, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(lstm_output)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)

    x2 = Conv1D(filters, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(lstm_output)
    avg_pool3_gru = GlobalAveragePooling1D()(x2)
    max_pool3_gru = GlobalMaxPooling1D()(x2)

    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru])
    x = BatchNormalization()(x)
    x = Dropout(dropout)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dropout)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(num_labels, activation = "sigmoid")(x)
    model = Model(inputs=language_model.input, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])

    return model

if __name__ == '__main__':
    import pandas as pd

    # 1. Set up language model.
    language_corpus = pickle.load(open('assets/wikitext-103/wikitext-103.corpus','rb'))
    num_words = len(language_corpus.word2idx) + 1
    language_model = build_fast_language_model(num_words, embedding_size=300, dropouti=0.2, rnn_sizes=(1024, 512),
                                               use_qrnn=False, use_gpu=True)
    #language_model.load_weights('../input/keras-ulmfit/test2.hdf5')

    classification_model = language_classification_model(language_model, num_labels=4, lr=0.1,
                                                         lr_d=0.0, kernel_size1=3, kernel_size2=2,
                                                         dense_units=128, dropout=0.1, filters=32)
    classification_model.summary()

    # 2. Set up train/test data
    train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
    test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")

    X_train = tokenize(train, maxlen=50)
    X_test = tokenize(test, maxlen=50)