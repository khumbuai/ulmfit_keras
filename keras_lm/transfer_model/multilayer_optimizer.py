"""
Implementation of Discriminative fine-tuning as explained in https://arxiv.org/pdf/1801.06146.pdf

Adatped from https://github.com/brunoklein99/srcnn/blob/5e874eb161d4d27cfdb6ac9b2196b3ad154fc672/LRMultiplierSGD.py#L46
"""

import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer


class LRMultiplierSGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Implements the Discriminative fine-tuning as explained in https://arxiv.org/pdf/1801.06146.pdf
    Current implementation is such that it is assumed that each layer consists of two update parameters,
    namely weights and bias.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    Example use:
    opt = LRMultiplierSGD(lr=1e-4, momentum=0.9, multipliers=[1, 1, 1, 1, 0.1, 0.1])
    model.compile(optimizer=opt, loss=mean_squared_error, metrics=[psnr])

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., discrimative_decay=1 / 2.6,
                 nesterov=False, **kwargs):
        super(LRMultiplierSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.discrimative_decay = discrimative_decay
        self.initial_decay = decay
        self.nesterov = nesterov


    def _set_up_discriminative_fine_tuning(self, params):
        """
        Sets up the decay for different layers in the network. The decay is calculated according to
        decay[layer_(i + 1)] = decay[layer_i] * decay.
        :param params:
        :return:
        """
        # Example of param.name:
        # lstm_1/kernel:0
        # lstm_1/recurrent_kernel:0
        # lstm_1/bias:0
        # lstm_2/kernel:0
        # lstm_2/recurrent_kernel:0
        # lstm_2/bias:0
        # dense_1/kernel:0
        # dense_1/bias:0
        names = [param.name.split('/')[0] for param in params]
        print(names)
        number_of_layers = len(set(names))

        def list_to_depth(names):
            """
            ['lstm_1', 'lstm_1', 'lstm_1', 'lstm_2', 'lstm_2', 'lstm_2', 'dense_1', 'dense_1']
            goes to
            [1, 1, 1, 2, 2, 2, 3, 3]
            :param names:
            :return:
            """
            # TODO: refactor
            layer_depths = [1]
            layer_depth = 1
            for i, name in enumerate(names[1:]):
                if name != names[i]:
                    layer_depth += 1
                layer_depths.append(layer_depth)
            return layer_depths

        layer_depths = list_to_depth(names)

        layer_decay = [K.variable(self.discrimative_decay ** (number_of_layers - depth)) for depth in layer_depths]
        return layer_decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """

        :param loss:
        :param list[tf.Variable] params: list of tensorflow weights and biases
        :return:
        """
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        # Discriminative fine-tuning:
        # each layer is updated according to the idea from https://arxiv.org/pdf/1801.06146.pdf, after eq(2)

        # The learning rate of the uppermost trainable layer is decreased by discriminative_fine_tuning
        # len(params) // 2 is (approximately) the number of layers on the model -> weight+ bias per each layer
        discriminative_fine_tuning = self._set_up_discriminative_fine_tuning(params)

        for i, (p, g, m) in enumerate(zip(params, grads, moments)):

            v = self.momentum * m - (lr * discriminative_fine_tuning[i]) * g  # velocity

            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(LRMultiplierSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    import numpy as np

    model = Sequential()
    model.add(LSTM(4, input_shape=(8, 1), return_sequences=True))
    model.add(LSTM(4))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    opt = LRMultiplierSGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=opt, loss='mse')
    model.summary()

    x = np.random.rand(1000, 8, 1)
    y = np.random.rand(1000, 1)

    model.fit(x, y, batch_size=64, epochs=2)
