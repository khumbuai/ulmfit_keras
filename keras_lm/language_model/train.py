import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import pickle
import numpy as np

from keras_lm.language_model.model import build_language_model
from keras_lm.preprocessing.batch_generators import BatchGenerator
from keras_lm.utils.utils import LoadParameters


def evaluate_model(model, word2idx, test_sentence, num_predictions=5):
    """
    Visual preidictions of the language model. The test_sentence is appended with num_predictions words,
    which are predicted as the next words from the model.
    :param str test_sentence:
    :param int num_predictions:
    :return: None
    """

    idx2word = {i:w for w,i in word2idx.items()}
    test_sentence = test_sentence.split()
    encoded_sentence = [word2idx[w] for w in test_sentence]

    for i in range(num_predictions):
        X = np.reshape(encoded_sentence, (1, len(encoded_sentence)))

        pred = model.predict(X)
        answer = np.argmax(pred, axis=2)

        predicted_idx = answer[0][-2]
        encoded_sentence.append(predicted_idx)

    print(' '.join([idx2word[i] for i in encoded_sentence]))


if __name__ == '__main__':

    # 1. Load parameters from config.yaml file
    params = LoadParameters()
    WIKIPEDIA_CORPUS_FILE = params.params['wikipedia_corpus_file']
    LANGUAGE_MODEL_WEIGHT = params.params['language_model_weight']
    LANGUAGE_MODEL_PARAMS = params.params['lm_params']

    epochs = params.params['lm_epochs']
    batch_size = params.params['lm_batch_size']
    valid_batch_size = params.params['lm_valid_batch_size']
    seq_len = params.params['lm_seq_len']

    # 2. Load Corpus
    corpus = pickle.load(open(WIKIPEDIA_CORPUS_FILE, 'rb'))

    train_gen = BatchGenerator(corpus.train, batch_size, 'normal', seq_len, modify_seq_len=True)
    valid_gen = BatchGenerator(corpus.valid, valid_batch_size, 'normal', seq_len, modify_seq_len=True)

    K.clear_session()
    num_words = len(corpus.word2idx) + 1

    model = build_language_model(num_words, **LANGUAGE_MODEL_PARAMS)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'],
                  optimizer=Adam(lr=3e-4, beta_1=0.8, beta_2=0.99))

    model.summary()

    callbacks = [EarlyStopping(patience=5),
                 ModelCheckpoint(LANGUAGE_MODEL_WEIGHT, save_weights_only=True)]

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(corpus.train) // (seq_len * batch_size),
                                  epochs=epochs,
                                  validation_data=valid_gen,
                                  validation_steps=len(corpus.valid) // (seq_len * batch_size),
                                  callbacks=callbacks,
                                  )

    evaluate_model(model, corpus.word2idx, 'i feel sick and go to the ')
