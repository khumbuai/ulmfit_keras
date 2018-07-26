from keras.layers import Lambda, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, BatchNormalization, Dense, Dropout
from keras.models import Model


def build_classification_model(language_model, num_labels, dense_units=128, dropout=0.1):
    """
    Transfer model for language classification. Implementation of the transfer model as explained in
    https://arxiv.org/abs/1801.06146.
    :param language_model:
    :param lr:
    :param lr_d:
    :param kernel_size1:
    :param kernel_size2:
    :param dense_units:
    :param dropout:
    :param filters:
    :return:
    """
    rnn_output = language_model.get_layer('final_rnn_layer').output

    avg_pool = GlobalAveragePooling1D()(rnn_output)
    max_pool = GlobalMaxPooling1D()(rnn_output)
    last_rnn_output = Lambda(lambda x: x[:, -1, :])(rnn_output)

    x = concatenate([avg_pool, max_pool, last_rnn_output])

    x = Dense(dense_units, activation='relu') (x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = Dense(dense_units, activation='relu') (x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = Dense(num_labels, activation = "sigmoid")(x)

    model = Model(inputs=language_model.input, outputs=x)

    return model


if __name__ == '__main__':
    from keras_lm.language_model.model import build_language_model
    language_model = build_language_model(num_words=100)
    language_model.summary()

    classification_model = build_classification_model(language_model, num_labels=5, dense_units=128, dropout=0.1)
    classification_model.summary()
