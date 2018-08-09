import os
import pickle
from collections import Counter

from tqdm import tqdm
import re
import html

from fastai.text import partition_by_cores, Tokenizer
import spacy
import sys


class Missingdict(dict):
    """
    Own implementation of defaultdict, which does not add a new key: value pair for keys which were not in the dictionary
    before lookup.
    """
    def __init__(self, *args,  default_value=None, **kwargs,):
        super(Missingdict, self).__init__(*args, **kwargs)
        self.default_value = default_value

    def __missing__(self, key):
        return self.default_value


class Corpus(object):
    """
    Loads the train.txt, valid.txt and test.txt files and tokenizes them.
    The class then contains the train, valid and test corpus (with word2idx applied), as well as the word2idx dictionary.
    """

    def __init__(self, path, lang='en', max_vocab=30000, min_freq=10, word2idx=None):
        try:
            spacy.load(lang)
        except OSError:
            print(f'spacy tokenization model is not installed for {lang}.')
            lang = lang if lang in ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl'] else 'xx'
            print(f'Command: python -m spacy download {lang}')
            sys.exit(1)

        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.lang = lang
        self.re1 = re.compile(r'  +')

        self.train_txt_path = os.path.join(path, 'train.txt')  # train.txt has 1 801 349 lines
        self.valid_txt_path = os.path.join(path, 'valid.txt')
        self.test_txt_path = os.path.join(path, 'test.txt')

        tokenized_train = self.tokenize(self.train_txt_path)
        if not word2idx:
            # Create word2idx from training corpus
            self.word2idx = self.create_word2idx(tokenized_train)
        else:
            self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.train = [self.word2idx[word] for word in tokenized_train]
        del tokenized_train

        tokenized_valid = self.tokenize(self.valid_txt_path)
        self.valid = [self.word2idx[word] for word in tokenized_valid]
        del tokenized_valid

        tokenized_test = self.tokenize(self.test_txt_path)
        self.test = [self.word2idx[word] for word in tokenized_test]
        del tokenized_test

        self.word2idx = dict(word2idx)

    def fixup(self, x):
        """
        Fixes some observed weird tokens
        :param x:
        :return:
        """
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
            ' @-@ ','-').replace('\\', ' \\ ')
        return self.re1.sub(' ', html.unescape(x))

    def create_word2idx(self, tokenized_corpus):
        freq = Counter(tokenized_corpus)
        print(freq.most_common(25))

        idx2word = [o for o, c in freq.most_common(self.max_vocab) if c > self.min_freq]
        idx2word.insert(0, '_pad_')
        idx2word.insert(0, '_unk_')

        word2idx = {v: k for k, v in enumerate(idx2word)}
        word2idx = Missingdict(word2idx, default_value=0)

        return word2idx

    def tokenize(self, path):
        """
        Tokenizes a text file. Every 50000 lines of a file, the text will be tokenized with
        fast.ai's tokenizer.
        :param path:
        :param num_tokens:
        :return:
        """
        tokenized_corpus = []
        text = ''
        with open(path, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                line = self.fixup(line)
                text += 'BOS' + line + 'EOS'
                if i % 50000 == 0 and i > 0:
                    tokenized_corpus += self._apply_tokenizer(text)
                    text = ''
        tokenized_corpus += self._apply_tokenizer(text)

        return tokenized_corpus

    def _apply_tokenizer(self, text):
        # Tokenizer creates a list, containing a list of each word pairs, such as
        #tokens = [[' ', 't_up', 'bos'], ['senjō'], ['no'], [' ', 't_up', 'eos']]
        tokens = Tokenizer(lang=self.lang).proc_all_mp(partition_by_cores(text.split()))
        # Flatten nested list
        return [word for sublist in tokens for word in sublist]


def create_corpus(PYTORCH_IDX2WORD_FILEPATH, WIKI103_FOLDER):
    # 1. Load words from pretrained pytorch language model
    # The itos_wt103.pkl file can be found on http://files.fast.ai/models/wt103/
    with open(PYTORCH_IDX2WORD_FILEPATH, 'rb') as f:
        words = pickle.load(f)
    word2idx = {word: idx for idx, word in enumerate(words)}
    word2idx = Missingdict(word2idx, default_value=0)

    # 2. Create corpus from Wiki103 files
    corpus = Corpus(WIKI103_FOLDER, word2idx=word2idx)

    to_save = [corpus.train, corpus.valid, corpus.test, corpus.word2idx, corpus.idx2word]
    with open(os.path.join(WIKI103_FOLDER, 'wikitext-103.corpus'), 'wb') as f:
        pickle.dump(to_save, f)


if __name__ == '__main__':
    from keras_lm.utils.utils import LoadParameters

    params = LoadParameters()
    WIKI103_FOLDER = params.params['wiki103_text_folder']
    PYTORCH_IDX2WORD_FILEPATH = params.params['pytorch_idx2word_filepath']

    create_corpus(PYTORCH_IDX2WORD_FILEPATH, WIKI103_FOLDER)

