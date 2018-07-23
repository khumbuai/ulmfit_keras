import logging
import os
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm


class Dictionary(object):
    def __init__(self):
        self.unk_token = '<unk>'
        self.max_vocab = 100000
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.max_vocab = 100000
        self.word2idx = {}
        self.num_tokens = [0, 0, 0]
        self.unk_token = '<unk>'
        #self.dictionary = Dictionary()
        self.train_txt_path = os.path.join(path, 'train.txt')
        self.valid_txt_path = os.path.join(path, 'valid.txt')
        self.test_txt_path = os.path.join(path, 'test.txt')
        self.get_vocab()
        self.idx2word = {i:w for w,i in self.word2idx.items()}
        self.train = self.tokenize(self.train_txt_path,self.num_tokens[0])
        self.valid = self.tokenize(self.valid_txt_path,self.num_tokens[1])
        self.test = self.tokenize(self.test_txt_path,self.num_tokens[2])

    def get_vocab(self):
        counter = Counter()
        for i,path in enumerate([self.train_txt_path, self.valid_txt_path, self.test_txt_path]):
            with open(path, 'r') as f:
                for line in tqdm(f):
                    words = line.split() + ['<eos>']
                    self.num_tokens[i] += len(words)
                    counter.update(words)
        words = [w[0] for w in counter.most_common(self.max_vocab)]
        self.word2idx = {w:i for i,w in enumerate(words)}

    def tokenize(self, path, num_tokens):
        #"Tokenizes a text file."
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.repeat(-1,num_tokens)
            token = 0
            for line in tqdm(f):
                words = line.split() + ['<eos>']
                for word in words:
                    try:
                        ids[token] = self.word2idx[word]
                    except KeyError:
                        ids[token] = self.word2idx[self.unk_token]
                    token += 1

        return ids


def get_wikitext103_corpus():
    '''
    Get's the wikitext-103 corpus. Corpus will be created, if not already saved.
    :return: Instance of Corpus, containing the wikitext-103 dataset
    :rtype Corpus:
    '''
    #if not os.path.exists(corpus_directory):
    #    os.makedirs(corpus_directory)
    #    corpus_files={'https://drive.google.com/open?id=1dmOWRaDm0R6dSN2rwPnL7l4Ij2EDDhk6': 'train.txt',
    #                  'https://drive.google.com/open?id=1fSA2yTMMO3y6bFtV9JH9RQOtwfaEAIU5': 'test.txt',
    #                  'https://drive.google.com/open?id=1SNlsj37tfMwyMwhiNg795KjnGKhrQNjO': 'valid.txt'}
    #    for url, filename in corpus_files.items():
    #        maybe_download(filename=filename,
    #                       source_url=url,
    #                       work_directory=corpus_directory)

    if not os.path.exists('assets/wikitext-103/'):
        corpus = Corpus('assets/wikitext-103/')
        pickle.dump(corpus,open('assets/wikitext-103/wikitext-103.corpus','wb'))
        logging.info('Corpus successfully created')

    return pickle.load(open('assets/wikitext-103/wikitext-103.corpus','rb'))


if __name__ == '__main__':
    get_wikitext103_corpus()
