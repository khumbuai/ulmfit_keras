import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras_wiki_lm.preprocessing.data import get_wikitext103_corpus
from keras_wiki_lm.preprocessing import utils
from keras_wiki_lm.model.language_model import build_language_model


def train_language_model(corpus, epochs=5):
    eval_batch_size = 10
    test_batch_size = 1
    batch_size = 64
    SEQ_LEN = 50

    train_data = utils.batchify(corpus.train, batch_size)
    val_data = utils.batchify(corpus.valid, eval_batch_size)
    #test_data = batchify(corpus.test, test_batch_size)

    train_gen = utils.data_gen2(train_data, SEQ_LEN, batch_size=batch_size)
    valid_gen = utils.data_gen2(val_data, SEQ_LEN, batch_size=eval_batch_size)

    num_words = len(corpus.word2idx) +1
    model = build_language_model(num_words)
    model.summary()

    early_stop = EarlyStopping(patience=2)
    check_point = ModelCheckpoint('model.hdf5', save_weights_only=True)
    model = build_language_model(num_words)
    model.fit_generator(train_gen,
                        steps_per_epoch=train_data.shape[0]//SEQ_LEN,
                        epochs=epochs,
                        validation_data=valid_gen,
                        validation_steps=val_data.shape[0]//SEQ_LEN,
                        callbacks=[early_stop,check_point])

    #model.evaluate_generator(valid_gen, steps=val_data.shape[0]//SEQ_LEN, verbose=True)

    test_sentence ='i am sick . hence i go to the'.split()
    encoded_sentence = [corpus.word2idx[w] for w in test_sentence]
    for i in range(5):
        pred = model.predict(encoded_sentence)
        answer = np.argmax(pred,axis = 2)
        encoded_sentence.append([a[0] for a in answer][-1])

    print(' '.join([corpus.idx2word[i[0]] for i in answer]))
    print(' '.join([corpus.idx2word[i] for i in encoded_sentence]))
    return model

if __name__ == '__main__':
    train_language_model(corpus=get_wikitext103_corpus())