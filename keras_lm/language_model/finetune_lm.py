"""
Fintunes a pretrained language model on a new language corpus.
"""

import keras.backend as K
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras_lm.transfer_model.multilayer_optimizer import LRMultiplierSGD
from keras_lm.language_model.model import build_language_model
from keras_lm.preprocessing.batch_generators import BatchGenerator
from keras_lm.language_model.train import evaluate_model

import pickle
import numpy as np
from copy import deepcopy
import os

UNKNOWN_TOKEN = '<unk>'


def read_df(df):
    """
    Maps all entries in df.text into one single list.
    :param df:
    :return:
    """
    texts = [text for text in df.text]
    texts = ' '.join(texts)
    return texts


def update_word2idx(word_list, word2idx):
    '''
    Updates the word2idx dictionary with all words which are in the word_list, but do not appear
    in the word2idx.keys()
    :param words:
    :param word2idx:
    :return:
    '''
    words_not_in_corpus = list(set(word_list) - set(word2idx.keys()))
    additional_word2idx = {words_not_in_corpus[i]: len(word2idx) + i
                           for i in range(len(words_not_in_corpus))
                           }
    word2idx.update(additional_word2idx)

    return word2idx


def update_embedding_weights(embedding_weights, num_words_not_in_corpus):
    embedding_weights = embedding_weights[0]  # shape (len(language_corpus.word2idx), embedding_size)
    mean_embedding_vector = embedding_weights.mean(axis=0)
    embedding_weights = np.append(embedding_weights, [mean_embedding_vector for _ in range(num_words_not_in_corpus)], axis=0)
    return embedding_weights


def update_language_model(language_model, num_words_not_in_corpus, **kwargs):
    weights = language_model.get_weights()

    old_embedding_weights = language_model.get_layer('embedding').get_weights()  # shape (len(language_corpus.word2idx), embedding_size)
    new_embedding_weights = update_embedding_weights(old_embedding_weights, num_words_not_in_corpus)

    # TODO infer the model parameters (rnn_sizes) from the old language_model
    weights[0] = new_embedding_weights

    updated_language_model = build_language_model(num_words=new_embedding_weights.shape[0],
                                                  embedding_size=new_embedding_weights.shape[1],
                                                  **kwargs)

    updated_language_model.set_weights(weights)

    return updated_language_model


if __name__ == '__main__':
    import pandas as pd

    CORPUS_FILEPATH = 'assets/wikitext-103'
    WIKITEXT_CORPUS_FILE = os.path.join(CORPUS_FILEPATH, 'wikitext-103.corpus')
    Word2IDX_FILE = os.path.join(CORPUS_FILEPATH, 'fintuned_word2idx.p')

    FINETUNED_CORPUS_FILEPATH = 'assets/finetuned_corpus/'

    WEIGTHS_FILEPATH = 'weights/language_model.hdf5'
    FINETUNED_WEIGTHS_FILEPATH = 'weights/language_model_finetuned.hdf5'

    LANGUAGE_MODEL_PARAMS = {'embedding_size': 400, 'rnn_size': (1150, 1150), 'use_gpu':True,
                             'dropout':0.1, 'tie_weights':True, 'use_qrnn':False, 'only_last': False
                            }

    batch_size = 64
    seq_length = 50

    # 1. Initialize pretrained language model.
    K.clear_session()
    wikitext_corpus = pickle.load(open(WIKITEXT_CORPUS_FILE, 'rb'))
    word2idx = wikitext_corpus.word2idx

    num_words = len(wikitext_corpus.word2idx) +1

    language_model = build_language_model(num_words, **LANGUAGE_MODEL_PARAMS)
    language_model.summary()
    #language_model.load_weights(WEIGTHS_FILEPATH)

    # 2. Open target training dataset. We assume that the dataframes contains the already tokenized sentences.
    train_df = pd.read_csv(os.path.join(FINETUNED_CORPUS_FILEPATH, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(FINETUNED_CORPUS_FILEPATH, 'train.csv'))

    train_text = read_df(train_df)
    valid_text = read_df(valid_df)

    text = train_text + valid_text

    # 3. Add new words to the word2idx dictionary and update the language model.
    num_words_not_in_corpus = len(set(text) - set(wikitext_corpus.word2idx.keys()))
    word2idx = update_word2idx(text, word2idx)
    language_model = update_language_model(language_model, num_words_not_in_corpus, **LANGUAGE_MODEL_PARAMS)

    # 3. Prepare training and validation data
    train = [word2idx[word] for word in train_text]
    valid = [word2idx[word] for word in valid_text]

    # 4. Finetune model

    train_gen = BatchGenerator(train, batch_size=batch_size, model_description='normal', modify_seq_len=True).batch_gen(seq_length)
    valid_gen = BatchGenerator(valid, batch_size=batch_size, model_description='normal', modify_seq_len=True).batch_gen(seq_length)

    callbacks = [EarlyStopping(patience=5),
                 ModelCheckpoint(FINETUNED_WEIGTHS_FILEPATH, save_weights_only=True)]

    language_model.compile(loss=sparse_categorical_crossentropy,
                           metrics=['sparse_categorical_accuracy'],
                           optimizer=LRMultiplierSGD(lr=0.2 * 3e-4, momentum=0., decay=0., nesterov=False)
                           )

    language_model.fit_generator(train_gen,
                                 steps_per_epoch=len(train)//(seq_length * batch_size),
                                 epochs=20,
                                 validation_data=valid_gen,
                                 validation_steps=len(valid)//(seq_length * batch_size),
                                 callbacks=callbacks,
                                 )

    evaluate_model(language_model, wikitext_corpus.word2idx, 'i feel sick and go to the', num_predictions=5)

    # 5. Save word2idx dictionary
    with open(Word2IDX_FILE, 'wb') as f:
        pickle.dump(wikitext_corpus.word2idx, f)
