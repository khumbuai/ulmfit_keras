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
