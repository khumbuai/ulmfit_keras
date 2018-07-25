# adatped from https://github.com/brunoklein99/srcnn/blob/5e874eb161d4d27cfdb6ac9b2196b3ad154fc672/LRMultiplierSGD.py#L46

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

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(LRMultiplierSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
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
        discriminative_fine_tuning = K.variable((1/2.6) ** len(params) // 2)

        print((1/2.6) ** len(params))

        for i, (p, g, m) in enumerate(zip(params, grads, moments)):

            v = self.momentum * m - (lr * discriminative_fine_tuning) * g  # velocity

            # increase discriminative_fine_tuning for the next layer. params -> weight+ bias per each layer
            if i % 2 == 0 and i >1:
                discriminative_fine_tuning = discriminative_fine_tuning * K.variable(2.6)

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
    from keras.layers import Dense
    import numpy as np

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = LRMultiplierSGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=opt, loss='mse')
    model.summary()

    x = np.random.rand(1000, 8)
    y = np.random.rand(1000, 1)

    model.fit(x, y, batch_size=64, epochs=2)
