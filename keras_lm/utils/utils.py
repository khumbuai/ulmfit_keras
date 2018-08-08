import numpy as np
import csv
import urllib3
import os
import pathlib
import yaml
from attrdict import AttrDict
import pandas as pd

from fastai.text import Tokenizer
from fastai.text import partition_by_cores

CONFIG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'config.yaml')


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class LoadParameters(_Borg):
    def __init__(self, fallback_file=CONFIG_PATH):
        _Borg.__init__(self)

        self.fallback_file = fallback_file
        self.params = self._read_yaml()

    def _read_yaml(self):
        with open(self.fallback_file) as f:
            config = yaml.load(f)
        return AttrDict(config)



def preprocess_imdb_sentiments(root_directory):
    """
    Transforms the IMDB sentiment dataset into a dataframe.

    IMBD dataset can be found on http://ai.stanford.edu/~amaas/data/sentiment/
    :param root_directory: Root directory containing the train and test folders
    :return:
    """

    classes = ['neg', 'pos']

    def get_texts(path):
        texts,labels = [],[]
        for idx, label in enumerate(classes):
            for fname in os.listdir(os.path.join(path, label)):
                if fname.endswith('.txt'):
                    with open(os.path.join(os.path.join(path, label), fname), 'r') as text:
                        texts.append(text.read())
                        labels.append(idx)

        texts = Tokenizer.proc_all_mp(partition_by_cores(texts))
        texts = [' '.join(text) + '<eos>' for text in texts]

        return np.array(texts), np.array(labels)

    trn_texts, trn_labels = get_texts(os.path.join(root_directory, 'train'))
    val_texts, val_labels = get_texts(os.path.join(root_directory, 'test'))

    col_names = ['labels', 'text']

    np.random.seed(42)
    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))

    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]

    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]

    df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

    df_trn[df_trn['labels'] != 2].to_csv(os.path.join(root_directory, 'train.csv'), header=False, index=False)
    df_val.to_csv(os.path.join(root_directory, 'test.csv'), header=False, index=False)


def maybe_download(filename, source_url, work_directory):
    """Download the filename from the source url into the working directory if it does not exist."""
    if not os.path.exists(work_directory):
        os.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        urllib3.disable_warnings()
        with urllib3.PoolManager() as http:
            r = http.request('GET', source_url)
            with open(filepath, 'wb') as fout:
                fout.write(r.read())

        print('{} succesfully downloaded'.format(filename))
    return filepath


def write_file(file_path, text_path, num_tokens):
    total_num_tokens = 0
    print(f'Writing to {file_path}...')
    with open(file_path, 'w', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        with open(text_path, 'r') as f:
            for i, text in enumerate(f):

                writer.writerow([text])
                # f_out.write(text)

                # calculate approximate length based on tokens
                total_num_tokens += len(text.split())
                if total_num_tokens > num_tokens:
                    break
                if i % 10000 == 0:
                    print('Processed {:,} documents. Total # tokens: {:,}.'.format(i, total_num_tokens))
    print('{}. # documents: {:,}. # tokens: {:,}.'.format(file_path, i, total_num_tokens))


if __name__=='__main__':
    params = LoadParameters()
    print(params.params)
    print(type(params.params['lm_params']))

    preprocess_imdb_sentiments('/Users/macuni/Documents/aclImdb')
