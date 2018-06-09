# Keras Normalized Optimizers
[![Travis](https://travis-ci.org/titu1994/keras-normalized-optimizers.svg?branch=master&style=flat-square)](https://travis-ci.org/titu1994/keras-normalized-optimizers.svg?branch=master)

Keras wrapper class for Normalized Gradient Descent from [kmkolasinski/max-normed-optimizer](https://github.com/kmkolasinski/deep-learning-notes/tree/master/max-normed-optimizer), which can be applied to almost all Keras optimizers.

Partially implements [Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network](https://arxiv.org/abs/1707.04822) for all base Keras optimizers, and allows flexibility to choose any normalizing function. It does not implement adaptive learning rates however.

# Usage

## Pre-defined normalizations

There are several normalization functions available to the `NormalizedOptimizer` class which wraps another Keras Optimizer. The available normalization functions are : 

- l1 : `sum(abs(grad))`. L1 normalization (here called max-normalization).
- l2 : `sqrt(sum(square(grad)))`. L2 normalization (Frobenius norm) is the **default normalization**.
- l1_l2 : Average of `l1` and `l2` normalizations
- avg_l1 : `mean(abs(grad))`. Similar to L1 norm, however takes average instead of sum.
- avg_l2 : `sqrt(mean(square(grad)))`. Similar to L2 norm, however takes average instead of sum.
- avg_l1_l2 : Average of `avg_l1` and `avg_l2` normalizations.
- max : `max(abs(grad))`. Takes the maximum as the normalizer. Ensures largest gradient = 1.
- min_max : Average of `max(abs(grad))` and `min(abs(grad))`.
- std : Uses the standard deviation of the gradient as normalization.

```python
from keras.optimizers import Adam, SGD
from optimizer import NormalizedOptimizer

sgd = SGD(0.01, momentum=0.9, nesterov=True)
sgd = NormalizedOptimizer(sgd, normalization='l2')

adam = Adam(0.001)
adam = NormalizedOptimizer(adam, normalization='l2')
```

## Custom normalizations
Apart from the above normalizations, **it also possible to dynamically add more normalizers at run time**. The normalization function must take a single Tensor as input and output a normalized Tensor.

The class method `NormalizedOptimizer.set_normalization_function(func_name, normalization_fn)` can be used to register new normalizers dynamically.

However, care must be taken to register these custom normalizers prior to loading a Keras Model (ex : `load_model` will fail otherwise).

```python
from keras.optimizers import Adam
from optimizer import NormalizedOptimizer
from keras import backend as K

# dummy normalizer which is basically `avg_l1` normalizer
def dummy_normalization(grad):
    norm = K.mean(K.abs(grad)) + K.epsilon()
    normalized_grad = grad / norm
    return normalized_grad
    
# give the new normalizer a name
normalizer_name = 'mean`

NormalizedOptimizer.set_normalization_function(normalizer_name, dummy_normalization)

# now these models can be used just like before
adam = Adam(0.001)
adam = NormalizedOptimizer(adam, normalization=normalizer_name)

```

# Results
## Convex Optimization

We optimize the loss function : 

```
L(x) = 0.5 x^T Q x + b^T x
```
where `Q` is random positive-definite matrix, `b` is a random vector

### **Normalized SGD (NSGD)**
------------
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/simple_optm_sgd.png?raw=true" height=50%>

### **Normalized Adam (NADAM)**
------------
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/simple_optm_adam.png?raw=true" height=50%>

We also inspect how the initial choice of learning rate affects Normalized Adam for a convex optimization problem below.

------------
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/lr_dependency_adam.png?raw=true" height=50%>

## Deep MLP
Model is same as in the Tensorflow codebase [kmkolasinski/max-normed-optimizer](https://github.com/kmkolasinski/deep-learning-notes/tree/master/max-normed-optimizer)

```
* 30 dense layers of size 128.
* After each layer Batchnormalization is applied then dropout at level 0.2
* Small l2 regularization is added to the weights of the network
```
------------
### **Training Graph**
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/mlp_train.png?raw=true" height=50%>

------------
### **Testing Graph**
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/mlp_test.png?raw=true" height=50%>

## CIFAR-10

The implementation of the model is kept same as in the Tensorflow repository.

------------
### **Train Graph**
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/cifar_train.png?raw=true" height=50%>

------------
### **Testing Graph**
<img src="https://github.com/titu1994/keras-normalized-optimizers/blob/master/images/cifar_test.png?raw=true" height=50%>

# Requirements

- Keras 2.1.6+
- Tensorflow / Theano (CNTK not tested, but should work)
