import numpy as np
import csv
import urllib3
import os
import pathlib
import yaml
from attrdict import AttrDict

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
    #write_file('assets/val.csv','assets/wikitext-103/valid.txt',1000000)
    #write_file('assets/train.csv', 'assets/wikitext-103/train.txt', 1000000)
