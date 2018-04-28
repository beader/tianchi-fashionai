from keras import backend as K


def SquareNormalizedError(normalizer_indices):
    assert len(normalizer_indices) % 2 == 0
    ni = [n * 2 for n in normalizer_indices] + [n * 2 + 1 for n in normalizer_indices]

    def wrapper(y_true, y_pred):
        scaler = K.zeros_like(y_pred[:, 0])
        for i in range(0, len(ni), 2):
            scaler = scaler + K.square(y_true[:, ni[i]] - y_true[:, ni[i + 1]])
        non_zero_mask = K.cast(y_true > 0, 'float')
        d = K.sum(K.square(y_true - y_pred * non_zero_mask), axis=-1)
        d = d / K.sum(non_zero_mask, axis=-1)
        sne = K.mean(d / scaler, axis=-1)
        return sne
    wrapper.__name__ = 'sne'
    return wrapper


def NormalizedError(sne):
    def wrapper(y_true, y_pred):
        return K.sqrt(sne(y_true, y_pred))
    wrapper.__name__ = 'ne'
    return wrapper


def mse(y_true, y_pred):
    non_zero_mask = K.cast(y_true > 0, 'float')
    return K.mean(K.square(y_pred * non_zero_mask - y_true), axis=-1)
