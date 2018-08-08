import numpy as np


class BatchGenerator:

    def __init__(self, tokenized_text, batch_size, seq_len, modify_seq_len=True, only_last=False):
        """
        :param array tokenized_text: array of encoded text
        :param batch_size:
        :param bool only_last: Names the model, for which we need batches
        :param bool modify_seq_len: Determines, whether the sequence length should be randomly changed after each batch
        """
        self.tokenized_text = tokenized_text
        self.batch_size = batch_size
        self.modify_seq_len = modify_seq_len
        self.only_last = only_last
        self.seq_len = seq_len

        self.pos = 0

    @staticmethod
    def _random_modify_seq_len(seq_len):
        seq_len = seq_len if np.random.random() < 0.95 else seq_len / 2.
        # Prevent excessively small or negative sequence lengths
        return max(5, int(np.random.normal(seq_len, 5)))


    def get_sample(self, pos, batch_seq_len):
        """
        Returns one x, y pair
        :param pos:
        :param batch_seq_len:
        :return:
        """
        start = (pos + 1) % len(self.tokenized_text)
        end = (pos + 1 + batch_seq_len) % len(self.tokenized_text)
        if start > end:  # text sequence is exhausted -> roll over to the start again
            start = len(self.tokenized_text) - batch_seq_len
            end = len(self.tokenized_text)
           
        data = self.tokenized_text[start -1: end - 1]
        target = self.tokenized_text[end - 1] if self.only_last else self.tokenized_text[start:end]
        # need to expand the dimension, such that Y has shape (batch_size, seq_len, 1) (for normal model),
        # or (batch_size, 1) (for many_to_one model)
        # -> needed for sparse_categorical_crossentropy loss
        target = np.expand_dims(target, axis=1)

        return data, target

    def generate_one_batch(self, pos, batch_seq_len):
        """
        Batch is of size:
        X: (batch_size, seq_len) Y: (batch_size, seq_len, 1) for self.model_description == normal
        X: (batch_size, seq_len) Y: (batch_size, 1) for self.model_description == many_to_one
        X: (2, batch_size, seq_len) Y: (batch_size, seq_len) for self.model_description == fast
        :param int pos:
        :param int seq_len:
        :return:
        """
        X = []
        Y = []
        for i in range(self.batch_size):
            x, y = self.get_sample(pos, batch_seq_len)
            X.append(x)
            Y.append(y)
            pos = (pos + batch_seq_len) % len(self.tokenized_text)

        return np.array(X), np.array(Y)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generates a batch for training/validation
        :param seq_len:
        :return:
        """
        batch_seq_len = self.seq_len
        if self.modify_seq_len:
            batch_seq_len = self._random_modify_seq_len(batch_seq_len)

        X, Y = self.generate_one_batch(self.pos, batch_seq_len)
        self.pos = (self.pos + batch_seq_len * self.batch_size) % len(self.tokenized_text)

        return X, Y


if __name__ == '__main__':
    import pickle
    from keras_lm.utils.utils import LoadParameters

    # 1. Load parameters from config.yaml file
    params = LoadParameters()
    wikipedia_corpus_file = params.params['wikipedia_corpus_file']

    corpus = pickle.load(open(wikipedia_corpus_file, 'rb'))

    def visualize_batches(generator):
            for _ in range(4):
                X, Y = next(generator)
                print(X.shape)
                print(Y.shape)
                print('~~~~~~~~~~~')

    normal_batch_generator = BatchGenerator(corpus.train, batch_size=10, seq_len=50, modify_seq_len=False)
    many_to_one_batch_generator = BatchGenerator(corpus.train, batch_size=10, seq_len=50, modify_seq_len=False, only_last=True)

    print('Normal')
    visualize_batches(normal_batch_generator)
    print('Many to one')
    visualize_batches(many_to_one_batch_generator)
