from __future__ import division

from typing import Union, Callable
from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


def max_normalization(grad):
    """
    Uses the L-infinity norm to compute the normalized
    gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    grad_max = K.max(K.abs(grad))
    norm = grad_max + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def min_max_normalization(grad):
    """
    Uses the average of the Max and Min of the absolute
    values of the gradients to compute the normalized
    gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    grad_min = K.min(K.abs(grad))
    grad_max = K.max(K.abs(grad))
    norm = ((grad_max + grad_min) / 2.0) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def std_normalization(grad):
    """
    Uses the standard deviation of the gradient to compute
    the normalized gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    norm = K.std(grad) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def l1_normalization(grad):
    """
    Uses the L-1 norm of the gradient to compute the normalized
    gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    norm = K.sum(K.abs(grad)) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def l2_normalization(grad):
    """
    Uses the L-2 norm of the gradient to compute the normalized
    gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    normalized_grad = K.l2_normalize(grad)
    return normalized_grad


def l1_l2_normalization(grad):
    """
    Uses the average of the L-1 and L-2 norms of the gradient to
    compute the normalized gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    l1 = K.sum(K.abs(grad))
    l2 = K.l2_normalize(grad)
    norm = ((l1 + l2) / 2.) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def average_l1_normalization(grad):
    """
    Uses the average of the L-1 norm (instead of sum) to compute
    the normalized gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    norm = K.mean(K.abs(grad)) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def average_l2_normalization(grad):
    """
    Uses the average of the L-2 norm (instead of sum) to compute
    the normalized gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    norm = K.sqrt(K.mean(K.square(grad))) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


def average_l1_l2_normalization(grad):
    """
    Uses the average of the L-1 and L-2 norms (instead of the sum)
    to compute the normalized gradient.

    # Arguments:
        grad: gradient for a variable

    # Returns:
        The normalized gradient
    """
    l1_norm = K.mean(K.abs(grad))
    l2_norm = K.sqrt(K.mean(K.square(grad)))
    norm = ((l1_norm + l2_norm) / 2.) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad


class NormalizedOptimizer(optimizers.Optimizer):

    def __init__(self, optimizer: Union[str, optimizers.Optimizer], normalization: str = 'l2'):
        """
        Creates a wrapper for a Keras optimizer such that its gradients are
        normalized prior to computing the update ops.

        Since it is a wrapper optimizer, it must delegate all normal optimizer
        calls to the optimizer that it wraps.

        Note:
            This wrapper optimizer monkey-patches the optimizer it wraps such that
            the call to `get_gradients` will call the gradients of the
            optimizer and then normalize the list of gradients.

            This is required because Keras calls the optimizer's `get_gradients`
            method inside `get_updates`, and without this patch, we cannot
            normalize the gradients before computing the rest of the
            `get_updates` code.

        # Arguments:
            optimizer: Keras Optimizer or a string. All optimizers other
                than TFOptimizer are supported. If string, instantiates a
                default optimizer with that alias.

            normalization: string. Must refer to a normalization function
                that is available in this modules list of normalization
                functions. To get all possible normalization functions,
                use `NormalizedOptimizer.get_normalization_functions()`.

        # Raises
            ValueError: If an incorrect name is supplied for `normalization`,
                such that the normalization function is not available or not
                set using `NormalizedOptimizer.set_normalization_functions()`.

            NotImplementedError: If `optimizer` is of type `TFOptimizer`.
        """
        if optimizer.__class__.__name__ == 'TFOptimizer':
            raise NotImplementedError('Currently, TFOptimizer is not supported.')

        if normalization not in _NORMS:
            raise ValueError('`normalization` must be one of %s.\n' 
                             'Provided was "%s".' % (str(sorted(list(_NORMS.keys()))), normalization))

        self.optimizer = optimizers.get(optimizer)
        self.normalization = normalization
        self.normalization_fn = _NORMS[normalization]

        # patch the `get_gradients` call
        self._optimizer_get_gradients = self.optimizer.get_gradients
        self.optimizer.get_gradients = self.get_gradients

    def get_gradients(self, loss, params):
        """
        Compute the gradients of the wrapped Optimizer, then normalize
        them with the supplied normalization function.

        # Arguments:
            loss: Keras tensor with a single value.
            params: List of tensors to optimize

        # Returns:
            A list of normalized gradient tensors
        """
        grads = self._optimizer_get_gradients(loss, params)
        grads = [self.normalization_fn(grad) for grad in grads]
        return grads

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """
        Computes the update operations of the wrapped Optimizer using
        normalized gradients and returns a list of operations.

        # Arguments:
            loss: Keras tensor with a single value
            params: List of tensors to optimize

        # Returns:
            A list of parameter and optimizer update operations
        """
        self.optimizer.get_updates(loss, params)
        return self.updates

    def set_weights(self, weights):
        """
        Set the weights of the wrapped optimizer by delegation

        # Arguments:
            weights: List of weight matrices
        """
        self.optimizer.set_weights(weights)

    def get_weights(self):
        """
        Get the weights of the wrapped optimizer by delegation

        # Returns:
            List of weight matrices
        """
        return self.optimizer.get_weights()

    def get_config(self):
        """
        Updates the config of the wrapped optimizer with some meta
        data about the normalization function as well as the optimizer
        name so that model saving and loading can take place

        # Returns:
            dictionary of the config
        """
        # properties of NormalizedOptimizer
        config = {'normalization': self.normalization,
                  'optimizer_name': self.optimizer.__class__.__name__.lower()}

        # optimizer config
        optimizer_config = {'optimizer_config': self.optimizer.get_config()}
        return dict(list(optimizer_config.items()) + list(config.items()))

    @property
    def weights(self):
        return self.optimizer.weights

    @property
    def updates(self):
        return self.optimizer.updates

    @classmethod
    def from_config(cls, config):
        """
        Utilizes the meta data from the config to create a new instance
        of the optimizer which was wrapped previously, and creates a
        new instance of this wrapper class.

        # Arguments:
            config: dictionary of the config

        # Returns:
            a new instance of NormalizedOptimizer
        """
        optimizer_config = {'class_name': config['optimizer_name'],
                            'config': config['optimizer_config']}

        optimizer = optimizers.get(optimizer_config)
        normalization = config['normalization']

        return cls(optimizer, normalization)

    @classmethod
    def set_normalization_function(cls, name: str, func: Callable):
        """
        Allows the addition of new normalization functions adaptively

        # Arguments:
            name: string name of the normalization function
            func: callable function which takes in a single tensor and
                returns a single tensor (input gradient tensor and output
                normalized gradient tensor).
        """
        global _NORMS
        _NORMS[name] = func

    @classmethod
    def get_normalization_functions(cls):
        """
        Get the list of all registered normalization functions that can be
        used.

        # Returns:
            list of strings denoting the names of all of the normalization
            functions.
        """
        global _NORMS
        return sorted(list(_NORMS.keys()))


_NORMS = {
    'max': max_normalization,
    'min_max': min_max_normalization,
    'l1': l1_normalization,
    'l2': l2_normalization,
    'linf': max_normalization,
    'l1_l2': l1_l2_normalization,
    'std': std_normalization,
    'avg_l1': average_l1_normalization,
    'avg_l2': average_l2_normalization,
    'avg_l1_l2': average_l1_l2_normalization,
}

# register this optimizer to the global custom objects when it is imported
get_custom_objects().update({'NormalizedOptimizer': NormalizedOptimizer})
