import os
import pickle

import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from keras_lm.transfer_model.model import build_classification_model
from keras_lm.language_model.model import build_language_model
from keras_lm.transfer_model.multilayer_optimizer import LRMultiplierSGD


def train_classifiaction_model(classification_model, X_train, y_train, epochs_list, learning_rates_list):
    """
    Implementation of transfer training by unfreezing the layers step by step.
    :param classification_model:
    :param X_train:
    :param y_train:
    :param List[int] epochs_list: list containing the number of epochs for each unfreezing step
    :param List[int] learning_rates_list: list containing the learning rates for each unfreezing step
    :return:
    """
    for layer in classification_model.layers:
        layer.trainable = False

    for i, layer in enumerate(reversed(classification_model.layers)):
        layer.trainable = True

        classification_model.compile(loss=categorical_crossentropy, optimizer=LRMultiplierSGD(lr=learning_rates_list[i],
                                                                                               momentum=0., decay=0.,
                                                                                               nesterov=False))

        classification_model.fit(X_train, y_train,
                                 batch_size=None,
                                 epochs=epochs_list[i],
                                 verbose=1,
                                 callbacks=None,
                                 validation_split=0.1,
                                 shuffle=True)

    return classification_model


if __name__ == '__main__':
    CORPUS_FILEPATH = 'assets/wikitext-103/wikitext-103.corpus'
    TRAINING_DATA_FILEPATH = ''
    FINETUNED_WEIGTHS_FILEPATH = 'weights/language_model_finetuned.hdf5'
    NUMBER_OF_LABELS = 5

    # 1. Initialize pretrained language model.
    K.clear_session()
    wikitext_corpus = pickle.load(open(CORPUS_FILEPATH,'rb'))
    num_words = len(wikitext_corpus.word2idx) +1

    language_model = build_language_model(num_words, embedding_size=300, use_gpu=True)
    language_model.summary()
    #language_model.load_weights(FINETUNED_WEIGTHS_FILEPATH)

    # 2. Initialize classifiaction_model
    classification_model = build_classification_model(language_model, NUMBER_OF_LABELS, dense_units=128, dropout=0.1)

    # 3. Load X_train from pickle file
    with open(TRAINING_DATA_FILEPATH, 'rb') as f:
        X_train, y_train = pickle.load(f)

    # 4. Train classification model
    epochs_list = [1 for layer in classification_model.layers]
    learning_rates_list = [0.01 for layer in classification_model.layers]
    train_classifiaction_model(classification_model, X_train, y_train, epochs_list, learning_rates_list)