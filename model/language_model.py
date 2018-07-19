import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.layers import Input, CuDNNLSTM, LSTM, Embedding, Dense, CuDNNGRU, LSTM, TimeDistributed, Dropout
from keras.models import Model
from keras.optimizers import Adam

from keras_wiki_lm.model.tied_embeddings import TiedEmbeddingsTransposed
from keras_wiki_lm.model.qrnn import QRNN



def build_language_model(num_words, dropout=0.1, dropouth=0.3, dropouti=0.2, dropoute=0.1, wdrop=0.5,
                         tie_weights=True, use_qrnn=False, use_gpu=True):

    inp = Input(shape=(None,))
    emb = Embedding(num_words,300)
    emb_inp = emb(inp)
    emb_inp = Dropout(dropouti)(emb_inp)

    if use_qrnn:
        rnn = QRNN(1024, return_sequences=True, window_size=2)(emb_inp)
        rnn = QRNN(1024, return_sequences=True, window_size=1)(rnn)
        rnn = QRNN(300, return_sequences=True,window_size=1)(rnn)
    else:
        RnnUnit = CuDNNLSTM if use_gpu else LSTM
        rnn = RnnUnit(1024, return_sequences=True)(emb_inp)
        rnn = RnnUnit(1024, return_sequences=True)(rnn)
        rnn = RnnUnit(300, return_sequences=True)(rnn)

    if tie_weights:
        logits = TimeDistributed(TiedEmbeddingsTransposed(tied_to=emb,activation='softmax'))(rnn)
    else:
        logits = TimeDistributed(Dense(num_words, activation='softmax'))(rnn)
    out = Dropout(dropout)(logits)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr=3e-4, beta_1=0.8, beta_2=0.99), loss='sparse_categorical_crossentropy')
    model.summary()
    return model


def build_simple_language_model(num_words, embedding_size=300, use_gpu=False):
    RnnUnit = CuDNNLSTM if use_gpu else LSTM
    inp = Input(shape=(None,))
    emb = Embedding(num_words, embedding_size)
    emb_inp = emb(inp)
    rnn = RnnUnit(1024, return_sequences=True)(emb_inp)
    rnn = RnnUnit(1024, return_sequences=True)(rnn)
    rnn = RnnUnit(embedding_size, return_sequences=True)(rnn)
    #rnn = QRNN(256, return_sequences=True)(emb_inp)
    #rnn = QRNN(256)(rnn)
    #den = Dense(SEQ_LEN, activation='relu')(rnn)
    #out = TimeDistributed(Dense(num_words, activation='softmax'))(rnn)
    out = TimeDistributed(TiedEmbeddingsTransposed(tied_to=emb, activation='softmax'))(rnn)
    model = Model(inputs=inp, outputs=out)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


if __name__ == '__main__':
    model = build_language_model(num_words=10000)
    model.summary()

    # m2 = Model(inputs=inp, outputs=out)
    # #model = multi_gpu_model(model, gpus=2)
    # m2.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # m2.load_weights('test.hdf5')
    # m2.evaluate_generator(valid_gen, steps=val_data.shape[0]//SEQ_LEN, verbose=True)
    #
    # test_sentence ='i am sick and go'.split()
    # encoded_sentence = [corpus.word2idx[w] for w in test_sentence]
    # for i in range(5):
    #     pred = model.predict(encoded_sentence)
    #     answer = np.argmax(pred,axis = 2)
    #     encoded_sentence.append([a[0] for a in answer][-1])
    #
    # print(' '.join([corpus.idx2word[i[0]] for i in answer]))
    # print(' '.join([corpus.idx2word[i] for i in encoded_sentence]))
    #




    #output = TimeDistributed(Dense(opt.word_vocab_size, activation='softmax'))(x)