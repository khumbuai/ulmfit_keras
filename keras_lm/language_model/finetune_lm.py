"""
Fintunes a pretrained language model on a new language corpus.
"""

import os
import pickle

import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import sparse_categorical_crossentropy

from keras_lm.language_model.batch_generators import BatchGenerator
from keras_lm.language_model.model import build_language_model
from keras_lm.language_model.train import evaluate_model
from keras_lm.transfer_model.multilayer_optimizer import LRMultiplierSGD


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

    weights[0] = new_embedding_weights

    updated_language_model = build_language_model(num_words=new_embedding_weights.shape[0],
                                                  **kwargs)

    updated_language_model.set_weights(weights)

    return updated_language_model


if __name__ == '__main__':
    import pandas as pd
    from collections import Counter, defaultdict

    from keras_lm.utils.utils import LoadParameters

    # 1. Load parameters from config.yaml
    params = LoadParameters()

    PYTORCH_ITOS_FILEPATH = params.params['pytorch_idx2word_filepath']
    WEIGTHS_FILEPATH = params.params['language_model_weight']
    FINETUNED_WEIGTHS_FILEPATH = params.params['finetuned_language_model_weight']
    FINETUNED_WORD2IDX_FILEPATH = params.params['finetuned_word2idx_filepath']
    LANGUAGE_MODEL_PARAMS = params.params['lm_params']
    FINETUNED_CORPUS_FILEPATH = params.params['finetuned_corpus_filepath']

    batch_size = params.params['lm_batch_size']
    seq_length = params.params['lm_seq_len']

    # 2. Initialize pretrained language model.
    K.clear_session()
    with open(PYTORCH_ITOS_FILEPATH, 'rb') as f:
        words = pickle.load(f)

    word2idx = {word: idx for idx, word in enumerate(words)}

    word2idx = defaultdict(lambda: word2idx['_unk_'], word2idx)

    num_words = len(word2idx)

    language_model = build_language_model(num_words, **LANGUAGE_MODEL_PARAMS)
    language_model.summary()
    language_model.load_weights(WEIGTHS_FILEPATH)

    # 3. Open target training dataset. We assume that the dataframes contains the already tokenized sentences.
    train_df = pd.read_csv(os.path.join(FINETUNED_CORPUS_FILEPATH, 'train.csv'), names=['mood', 'text'])
    valid_df = pd.read_csv(os.path.join(FINETUNED_CORPUS_FILEPATH, 'test.csv'), names=['mood', 'text'])

    train_text = read_df(train_df)
    valid_text = read_df(valid_df)

    text = train_text + valid_text

    unique_words = [o for o, c in Counter(text).most_common(100000) if c > 10]

    # 4. Add new words to the word2idx dictionary and update the language model.
    num_words_not_in_corpus = len(set(unique_words) - set(word2idx.keys()))
    word2idx = update_word2idx(unique_words, word2idx)
    language_model = update_language_model(language_model, num_words_not_in_corpus, **LANGUAGE_MODEL_PARAMS)

    language_model.summary()

    # 5. Prepare training and validation data
    train = [word2idx[word] for word in train_text]
    valid = [word2idx[word] for word in valid_text]

    # 6. Finetune model
    train_gen = BatchGenerator(train, batch_size, seq_length, modify_seq_len=True)
    valid_gen = BatchGenerator(valid, batch_size, seq_length, modify_seq_len=True)

    callbacks = [EarlyStopping(patience=5),
                 ModelCheckpoint(FINETUNED_WEIGTHS_FILEPATH, save_weights_only=True)]

    language_model.compile(loss=sparse_categorical_crossentropy,
                           metrics=['sparse_categorical_accuracy'],
                           optimizer=LRMultiplierSGD(lr=0.2 * 3e-4, momentum=0., decay=0., nesterov=False)
                           )

    language_model.fit_generator(train_gen,
                                 steps_per_epoch=len(train) // (seq_length * batch_size),
                                 epochs=20,
                                 validation_data=valid_gen,
                                 validation_steps=len(valid) // (seq_length * batch_size),
                                 callbacks=callbacks,
                                 )

    evaluate_model(language_model, word2idx, 'i feel sick and go to the', num_predictions=5)

    # 7. Save word2idx dictionary
    with open(FINETUNED_WORD2IDX_FILEPATH, 'wb') as f:
        pickle.dump(word2idx, f)
