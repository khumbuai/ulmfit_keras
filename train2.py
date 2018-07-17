import data
from utils import batchify, get_batch, data_gen2
import numpy as np

import os
#from keras.utils import multi_gpu_model
from tqdm import tqdm
#import tensorflow as tf
import pickle
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.layers import Input, CuDNNLSTM, Embedding, Dense, CuDNNGRU, LSTM, TimeDistributed
from keras.models import Model, load_model
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tied_embeddings import TiedEmbeddingsTransposed
from qrnn import QRNN
#corpus = data.Corpus('assets/wikitext-103/')
#pickle.dump(corpus,open('wikitext-103.corpus','wb'))

corpus = pickle.load(open('wikitext-103.corpus','rb'))


eval_batch_size = 10
test_batch_size = 1
batch_size = 64
SEQ_LEN = 50
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, test_batch_size)

# save to temp

#data, targets = get_batch(val_data, 0, seq_len=SEQ_LEN)
#X = np.zeros(shape=(len(corpus.train)-SEQ_LEN -2,SEQ_LEN))
#y = np.zeros(shape=(len(corpus.train)-SEQ_LEN -2,))
#for i in tqdm(range(len(corpus.train)-SEQ_LEN -2)):
#    data, targets = get_batch(corpus.train,i,SEQ_LEN)
#    X[i] = data
#    y[i] = targets


train_gen = data_gen2(train_data,SEQ_LEN, batch_size=batch_size)
valid_gen = data_gen2(val_data,SEQ_LEN, batch_size=eval_batch_size)


num_words = len(corpus.word2idx) +1


inp = Input(shape=(None,))
emb = Embedding(num_words,300)
emb_inp = emb(inp)
rnn = CuDNNLSTM(1024, return_sequences=True)(emb_inp)
rnn = CuDNNLSTM(1024, return_sequences=True)(rnn)
rnn = CuDNNLSTM(300, return_sequences=True)(rnn)
#rnn = QRNN(256, return_sequences=True)(emb_inp)
#rnn = QRNN(256)(rnn)
#den = Dense(SEQ_LEN, activation='relu')(rnn)
#out = TimeDistributed(Dense(num_words, activation='softmax'))(rnn)
out = TimeDistributed(TiedEmbeddingsTransposed(tied_to=emb,activation='softmax'))(rnn)
model = Model(inputs=inp, outputs=out)
#model = multi_gpu_model(model, gpus=2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()


model.fit_generator(train_gen, steps_per_epoch=train_data.shape[0]//SEQ_LEN, epochs=5)
model.evaluate_generator(valid_gen, steps=val_data.shape[0]//SEQ_LEN, verbose=True)

m2 = Model(inputs=inp, outputs=out)
#model = multi_gpu_model(model, gpus=2)
m2.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
m2.load_weights('test.hdf5')
m2.evaluate_generator(valid_gen, steps=val_data.shape[0]//SEQ_LEN, verbose=True)

test_sentence ='i am sick and go'.split()
encoded_sentence = [corpus.word2idx[w] for w in test_sentence]
for i in range(5):
    pred = model.predict(encoded_sentence)
    answer = np.argmax(pred,axis = 2)
    encoded_sentence.append([a[0] for a in answer][-1])

print(' '.join([corpus.idx2word[i[0]] for i in answer]))
print(' '.join([corpus.idx2word[i] for i in encoded_sentence]))





#output = TimeDistributed(Dense(opt.word_vocab_size, activation='softmax'))(x)