import numpy as np


class BatchGenerator():

    def __init__(self, tokenized_text, batch_size, model_description, modify_seq_len=True):
        """
        :param array tokenized_text: array of encoded text
        :param batch_size:
        :param str model_description: Names the model, for which we need batches
        :param bool modify_seq_len: Determines, whether the sequence length should be randomly changed after each batch
        """
        self.tokenized_text = tokenized_text
        self.batch_size = batch_size
        self.modify_seq_len = modify_seq_len
        assert model_description in ['normal', 'many_to_one', 'fast'], 'Model not supported'
        self.model_description = model_description

    @staticmethod
    def _random_modify_seq_len(seq_len):
        seq_len = seq_len if np.random.random() < 0.95 else seq_len / 2.
        # Prevent excessively small or negative sequence lengths
        return max(5, int(np.random.normal(seq_len, 5)))

    def _reshape_batch_for_fast_language_model(self, X, Y):
        """
        Reshapes the (X, Y) batch such that it is valid input for the fast model architecture.
        :param array X: shape (batch_size, sequence_length)
        :param array Y: shape (batch_size, sequence_length, 1)
        :return:
        :rtype array:
        """
        Y = np.squeeze(Y)
        # !!!! Be carful not to convert the return value of [X, Y] to a numpy array, as it will then throw an error in the fit method!
        # https://stackoverflow.com/questions/46450184/keras-multiple-inputs-for-fit-generator-using-flow-from-directory

        # targets[0] -> dot product between rnn output and next word embedding.
        # target[1] -> dot product between current word embedding and next word embedding.
        # see build_fast_language_model in language models for model architecture
        targets = [np.ones_like(X), np.zeros_like(X)]

        # Implements Skip-gram negative sampling. Assumes that the corpus has enough words, such that
        # sampling Y is not probable
        negative_samples = np.random.choice(self.tokenized_text, X.shape)
        return [X, Y, negative_samples], targets

    def get_sample(self, pos, seq_len):
        """
        Returns one x, y pair
        :param pos:
        :param seq_len:
        :return:
        """
        start = (pos + 1) % len(self.tokenized_text)
        end = (pos + 1 + seq_len) % len(self.tokenized_text)
        if start > end:  # text sequence is exhausted -> roll over to the start again
            start = len(self.tokenized_text) - seq_len
            end = len(self.tokenized_text)
           
        data = self.tokenized_text[start -1: end - 1]
        target = self.tokenized_text[start:end] if self.model_description in ['normal', 'fast'] else self.tokenized_text[end - 1]
        # need to expand the dimension, such that Y has shape (batch_size, seq_len, 1) (for normal model),
        # or (batch_size, 1) (for many_to_one model)
        # -> needed for sparse_categorical_crossentropy loss
        target = np.expand_dims(target, axis=1)

        return data, target

    def generate_one_batch(self, pos, seq_len):
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
            x, y = self.get_sample(pos, seq_len)
            X.append(x)
            Y.append(y)
            pos = (pos + seq_len) % len(self.tokenized_text)

        X = np.array(X)
        Y = np.array(Y)
        if self.model_description == 'fast':
            X, Y = self._reshape_batch_for_fast_language_model(X, Y)

        return X, Y

    def batch_gen(self, seq_len):
        """
        Generates a batch for training/validation
        :param seq_len:
        :return:
        """
        pos = 0  # Points to the start of the batch
        while True:
            batch_seq_len = seq_len
            if self.modify_seq_len:
                batch_seq_len = self._random_modify_seq_len(batch_seq_len)

            X, Y = self.generate_one_batch(pos, batch_seq_len)
            yield X, Y

            pos = (pos + batch_seq_len * self.batch_size) % len(self.tokenized_text)


if __name__ == '__main__':
    import pickle
    corpus = pickle.load(open('assets/wikitext-103/wikitext-103.corpus','rb'))

    def visualize_batches(generator):
            generator = generator.batch_gen(seq_len=3)
            for _ in range(4):
                X, Y = next(generator)
                print(X)
                print(Y)
                print('~~~~~~~~~~~')

    normal_batch_generator = BatchGenerator(corpus.train, batch_size=2, model_description='normal', modify_seq_len=False)
    many_to_one_batch_generator = BatchGenerator(corpus.train, batch_size=2, model_description='many_to_one', modify_seq_len=False)
    fast_batch_generator = BatchGenerator(corpus.train, batch_size=2, model_description='fast', modify_seq_len=False)

    print('Normal')
    visualize_batches(normal_batch_generator)
    print('Many to one')
    visualize_batches(many_to_one_batch_generator)
    print('Fast')
    visualize_batches(fast_batch_generator)
