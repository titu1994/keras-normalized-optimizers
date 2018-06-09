from __future__ import print_function

import os
import pytest
import six
import numpy as np
from numpy.testing import assert_allclose

from keras.utils import test_utils
from keras import optimizers, Input
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Activation, Lambda
from keras.utils.np_utils import to_categorical
from keras import backend as K

from optimizer import NormalizedOptimizer

num_classes = 2


def get_test_data():
    np.random.seed(1337)
    (x_train, y_train), _ = test_utils.get_test_data(num_train=1000,
                                                     num_test=200,
                                                     input_shape=(10,),
                                                     classification=True,
                                                     num_classes=num_classes)
    y_train = to_categorical(y_train)
    return x_train, y_train


def _test_optimizer(optimizer, target=0.75):
    x_train, y_train = get_test_data()

    # if the input optimizer is not a NormalizedOptimizer, wrap the optimizer
    # with a default NormalizedOptimizer
    if optimizer.__class__.__name__ != NormalizedOptimizer.__name__:
        optimizer = NormalizedOptimizer(optimizer, normalization='l2')

    model = Sequential()
    model.add(Dense(10, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
    assert history.history['acc'][-1] >= target

    # Test optimizer serialization and deserialization.
    config = optimizers.serialize(optimizer)
    optim = optimizers.deserialize(config)
    new_config = optimizers.serialize(optim)
    assert config == new_config

    # Test weights saving and loading.
    original_weights = optimizer.weights

    model.save('temp.h5')
    temp_model = load_model('temp.h5')
    loaded_weights = temp_model.optimizer.weights
    assert len(original_weights) == len(loaded_weights)
    os.remove('temp.h5')

    # Test constraints.
    model = Sequential()
    dense = Dense(10,
                  input_shape=(x_train.shape[1],),
                  kernel_constraint=lambda x: 0. * x + 1.,
                  bias_constraint=lambda x: 0. * x + 2., )
    model.add(dense)
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.train_on_batch(x_train[:10], y_train[:10])
    kernel, bias = dense.get_weights()
    assert_allclose(kernel, 1.)
    assert_allclose(bias, 2.)


def _test_no_grad(optimizer):
    inp = Input([3])
    x = Dense(10)(inp)
    x = Lambda(lambda l: 1.0 * K.reshape(K.cast(K.argmax(l), 'float32'), [-1, 1]))(x)
    mod = Model(inp, x)
    mod.compile(optimizer, 'mse')
    with pytest.raises(ValueError):
        mod.fit(np.zeros([10, 3]), np.zeros([10, 1], np.float32), batch_size=10, epochs=10)


def test_sgd_normalized_from_string():
    sgd = NormalizedOptimizer('sgd', normalization='l2')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_max():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='max')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_min_max():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='min_max')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_l1():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='l1')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_l2():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='l2')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_l1_l2():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='l1_l2')
    _test_optimizer(sgd, target=0.45)
    _test_no_grad(sgd)


def test_sgd_normalized_std():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='std')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_average_l1():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='avg_l1')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_average_l2():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='avg_l2')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_sgd_normalized_average_l1_l2():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    sgd = NormalizedOptimizer(sgd, normalization='avg_l1_l2')
    _test_optimizer(sgd)
    _test_no_grad(sgd)


def test_rmsprop_normalized():
    _test_optimizer(optimizers.RMSprop())
    _test_optimizer(optimizers.RMSprop(decay=1e-3))


def test_adagrad_normalized():
    _test_optimizer(optimizers.Adagrad())
    _test_optimizer(optimizers.Adagrad(decay=1e-3))


def test_adadelta_normalized():
    _test_optimizer(optimizers.Adadelta())
    _test_optimizer(optimizers.Adadelta(decay=1e-3))


def test_adam_normalized():
    _test_optimizer(optimizers.Adam())
    _test_optimizer(optimizers.Adam(decay=1e-3))


def test_adamax_normalized():
    _test_optimizer(optimizers.Adamax())
    _test_optimizer(optimizers.Adamax(decay=1e-3))


def test_nadam_normalized():
    _test_optimizer(optimizers.Nadam())


def test_adam_amsgrad_normalized():
    _test_optimizer(optimizers.Adam(amsgrad=True))
    _test_optimizer(optimizers.Adam(amsgrad=True, decay=1e-3))


def test_clipnorm_normalized():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=0.5)
    sgd = NormalizedOptimizer(sgd, normalization='l2')
    _test_optimizer(sgd)


def test_clipvalue_normalized():
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    sgd = NormalizedOptimizer(sgd, normalization='l2')
    _test_optimizer(sgd)


def test_wrong_normalization():
    with pytest.raises(ValueError):
        NormalizedOptimizer('sgd', normalization=None)


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='TFOptimizer requires TF backend')
def test_tf_optimizer():
    with pytest.raises(NotImplementedError):
        import tensorflow as tf
        tf_opt = optimizers.TFOptimizer(tf.train.GradientDescentOptimizer(0.1))
        NormalizedOptimizer(tf_opt, normalization='l2')


def test_add_normalizer():
    def dummy_normalization(grad):
        norm = K.mean(K.abs(grad)) + K.epsilon()
        return norm

    func_name = 'dummy'

    # check that this function doesnt exist in the normalizers
    name_list = NormalizedOptimizer.get_normalization_functions()
    assert func_name not in name_list

    # add the function to the name list
    NormalizedOptimizer.set_normalization_function(func_name, dummy_normalization)

    # check if it exists in the name list now
    name_list = NormalizedOptimizer.get_normalization_functions()
    assert func_name in name_list

    # train a model on this new normalizer
    sgd = NormalizedOptimizer('sgd', normalization=func_name)
    _test_optimizer(sgd)
    _test_no_grad(sgd)


if __name__ == '__main__':
    pytest.main([__file__])
