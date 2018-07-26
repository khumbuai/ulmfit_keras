import keras.backend as K
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras_lm.transfer_model.multilayer_optimizer import LRMultiplierSGD
from keras_lm.language_model.model import build_language_model
from keras_lm.preprocessing.batch_generators import BatchGenerator
from keras_lm.language_model.train import evaluate_model

import pickle
import numpy as np
import pandas as pd
import os
from collections import defaultdict

UNKNOWN_TOKEN = '<unk>'


def generate_finetuned_data_from_df(df, word2idx):
    """
    Transforms a dataframe, containing the corpus items as values into one list of ints.
    The ints are the word indices from word2idx[word], where words not in the corpus are mapped to UNKNOWN_TOKEN.
    :param df:
    :param word2idx:
    :return:
    """
    word2idx = defaultdict(lambda: word2idx[UNKNOWN_TOKEN], word2idx)
    texts = np.array([[str(x) for x in y] for y in df.values])

    return [word2idx[word] for text in texts for word in text]


if __name__ == '__main__':

    CORPUS_FILEPATH = 'assets/wikitext-103/wikitext-103.corpus'
    FINETUNED_CORPUS_FILEPATH = 'assets/finetuned_corpus/train.p'

    WEIGTHS_FILEPATH = 'weights/language_model.hdf5'
    FINETUNED_WEIGTHS_FILEPATH = 'weights/language_model_finetuned.hdf5'

    batch_size = 64
    seq_length = 50

    # 1. Initialize pretrained language model.
    K.clear_session()
    wikitext_corpus = pickle.load(open(CORPUS_FILEPATH,'rb'))
    num_words = len(wikitext_corpus.word2idx) +1

    language_model = build_language_model(num_words, embedding_size=300, use_gpu=False)
    language_model.summary()
    #language_model.load_weights(WEIGTHS_FILEPATH)

    # 2. Prepare target training dataset.
    with open(FINETUNED_CORPUS_FILEPATH, 'rb') as f:
        train, valid = pickle.load(f)

    train_gen = BatchGenerator(train, batch_size=batch_size, model_description='normal', modify_seq_len=True).batch_gen(seq_length)
    valid_gen = BatchGenerator(valid, batch_size=batch_size, model_description='normal', modify_seq_len=True).batch_gen(seq_length)

    # 3. Finetune model
    callbacks = [EarlyStopping(patience=5),
                 ModelCheckpoint(FINETUNED_WEIGTHS_FILEPATH, save_weights_only=True)]

    language_model.compile(loss=sparse_categorical_crossentropy, optimizer=LRMultiplierSGD(lr=0.2 * 3e-4,
                                                                                           momentum=0., decay=0.,
                                                                                           nesterov=False))

    language_model.fit_generator(train_gen,
                                 steps_per_epoch=len(train)//(seq_length * batch_size),
                                 epochs=20,
                                 validation_data=valid_gen,
                                 validation_steps=len(valid)//(seq_length * batch_size),
                                 callbacks=callbacks,
                                 )

    evaluate_model(language_model, wikitext_corpus.word2idx, 'i feel sick and go to the', num_predictions=5)
