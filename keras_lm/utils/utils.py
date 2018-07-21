import numpy as np
import csv
import urllib3
import os


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
    write_file('assets/val.csv','assets/wikitext-103/valid.txt',1000000)
    write_file('assets/train.csv', 'assets/wikitext-103/train.txt', 1000000)
