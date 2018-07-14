import numpy as np

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

def data_gen(source, seq_len, batch_size = 1):
    while True:
        for i in range(len(source)):
            data, target = get_batch(source,i,seq_len)
            if batch_size == 1:
                data, target = np.expand_dims(data, axis=0), np.expand_dims(target, axis=0)
            else:
                data = data.T

            yield data, target
