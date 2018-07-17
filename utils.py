import numpy as np
import csv
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[:nbatch * batch_size]
    # Evenly divide the data across the bsz batches.
    data = np.reshape(data, newshape=(batch_size,-1)).T
    return data


def get_batch(source, i, seq_len=None):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1+seq_len]
    return data, target

def get_batch2(source, i, seq_len=None):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

def data_gen(source, bptt, batch_size):

    while True:
        for i in range(len(source)):
            bptt = bptt if np.random.random() < 0.95 else bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, target = get_batch(source,i,seq_len)
            if batch_size == 1:
                data, target = np.expand_dims(data, axis=0), np.expand_dims(target, axis=0)
            else:
                data = data.T

            yield data, target

def data_gen2(source, bptt, batch_size):
    i = 0
    while i < len(source):
            bptt = bptt if np.random.random() < 0.95 else bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            data, target = get_batch2(source,i,seq_len)
            if batch_size == 1:
                data, target = np.expand_dims(data, axis=0), np.expand_dims(target, axis=0)
            else:
                data = data.T
                target = np.expand_dims(target.T, axis=2)
            yield data, target
            i+= seq_len

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
