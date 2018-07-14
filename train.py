import data
from utils import batchify, get_batch, data_gen
import numpy as np
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
corpus = data.Corpus('assets/wikitext-103/')

eval_batch_size = 10
test_batch_size = 1
batch_size = 32
SEQ_LEN = 50
#train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, test_batch_size)

# save to temp

#data, targets = get_batch(val_data, 0, seq_len=SEQ_LEN)
#X = []
#y = []
#for i in range(len(corpus.valid)-SEQ_LEN -2):
#    data, targets = get_batch(corpus.valid,i,SEQ_LEN)
#    X.append(data)
#    y.append(targets)
#
#X = np.array(X)
#y = np.array(y)


valid_gen = data_gen(val_data,SEQ_LEN, batch_size=eval_batch_size)

num_words = len(corpus.dictionary.counter)

def build_model():

    inp = Input(shape=(None,))
    emb_inp = Embedding(num_words+1,100)(inp)
    rnn = LSTM(64)(emb_inp)
    out = Dense(num_words, activation='softmax')(rnn)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=sparse_categorical_crossentropy)
    model.summary()
    return model

model = build_model()

model.fit_generator(valid_gen, steps_per_epoch=200000,use_multiprocessing=True,
                    workers=6)


#model.fit(X,y)