"""A multi-layer perceptron for classification of MNIST handwritten digits with adam optimizer."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.flatten import flatten_func
from autograd.misc.optimizers import adam
from data import load_mnist


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

def next_batch_size(startsize, train_size, b, k):
    # exponentially schedule for minibatch size as described in P42.
    # k is the number of iterations.
    return int(min(startsize*2**(k/b), train_size))

def soft_max(x):
    # shape of x : (# of samples * dimension)
    e_x = np.exp((x.T - np.max(x, 1)).T)
    return (e_x.T / np.sum(e_x,1)).T

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_predict(outputs,layer_types):
    # return the final prediction of the neural network given the outputs of the last layer
    # so far only softmax function included, will include more types of last layer functions
    if layer_types[-1] == 'softmax':
        return outputs - logsumexp(outputs, axis=1, keepdims=True)  # a_l

def objective(outputs, params, targets, layer_types, L2_reg=1):
    # a_l is the predictions from the last layer
    # will modify is for different regularization and loss functions
    log_predictions = log_predict(outputs, layer_types)
    log_prior = L2_reg * l2_norm(params) # this is positive
    log_lik = np.sum(log_predictions * targets) # this is negative
    return log_prior - log_lik

def samplesoftmax(predictions):
    # predictions are inputs to the softmax layer
    c = np.cumsum(predictions, 1)
    s = npr.rand(predictions.shape[0])
    #comparison = np.tile(s, (predictions.shape[1], 1)).T <= c
    return np.diff(np.concatenate((np.zeros([1, predictions.shape[0]]).T,
                                   np.tile(s, (predictions.shape[1], 1)).T <= c),
                                  axis=1),axis=1)

# This will be replaced by the autograd function.
def one_forwardpass_and_one_backward_pass(params,inputs_minibatch, targets_minibatch, numlayers, layer_sizes, layer_types):
    # back_prop that keeps track of ai's and gi's and gives the gradient
    # for now, we don't break the minibatch into chunks
    a_inc = []
    a_inc.append(inputs_minibatch) # now a_inc includes the minibatch inputs
    g_inc = []
    inputs = inputs_minibatch
    #grad_a_l = grad(objective)  # the gradient of objective function with respect to a_l, the predictions.

    # forward prop
    for i in range(numlayers-1):
        outputs = np.dot(inputs, params[i][0]) + params[i][1] # outputs of the i-th layer: a_i
        if layer_types[i] == 'tanh':
            inputs = np.tanh(outputs) # activations of the i-th layer: a_i, will add more activations later
        a_inc.append(inputs)  # now a_inc includes the a_i_hom = [a_i, 1], i < l
    outputs = np.dot(inputs, params[numlayers-1][0]) + params[numlayers-1][1]  # outputs of the last layer

    # Backward Prop
    D_W_b = [[np.zeros([m, n]),  # matrix of gradient of weights
              np.zeros(n)]  # matrix of gradients of baises
             for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    if layer_types[-1] == 'softmax':
        predictions = soft_max(outputs)
        g = predictions - targets_minibatch # Gradient of objective function w.r.t. outputs of the last layer
    else:
        # 1. Last layer:
        # Gradient of objective function w.r.t. outputs of the last layer
        grad_last = grad(objective)
        g = grad_last(outputs, params, targets_minibatch, layer_types, L2_reg=1) # shape: (layer_sizes[l] * num of examples)

    D_W_b[-1][0] =  np.dot(a_inc[-1].T, g)# DW of the last layer ; shape: (layer_sizes[l-1], 10)
    D_W_b[-1][1] = np.sum(g,0) # Db of the last layer

    for i in range(1,numlayers):
        if layer_types[-i-1] == 'tanh':
            g = np.dot(g,params[-i][0].T) * (1 + a_inc[-i]) * (1 - a_inc[-i]) # note that a_inc[-1] is the second last layer
        D_W_b[-i-1][0] = np.dot(a_inc[-i-1].T, g)
        D_W_b[-i-1][1] = np.sum(g, 0)
        
    return D_W_b

def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

def batch_data(iter):
    idx = batch_indices(iter)
    return train_images[idx], train_labels[idx]

def batch_data_inc(minibatch_size, train_inputs , train_targets ):
    #exponentially increasing batch size

    idx = np.random.randint(train_size, size=minibatch_size)
    inputs_minibatch = train_inputs[idx, :]  # shape :(minibatch_size, dim of input)
    targets_minibatch = train_targets[idx, :]
    return inputs_minibatch, targets_minibatch


if __name__ == '__main__':
    # Model parameters
    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0
    numlayers = len(layer_sizes) - 1
    layer_types = ['tanh', 'tanh', 'softmax']
    # Training parameters
    param_scale = 0.1
    batch_size = 256
    num_epochs = 8
    step_size = 0.001
    print("Loading training data...")
    N, train_images, train_labels, test_images, test_labels = load_mnist()
    init_params = init_random_params(param_scale, layer_sizes)
    num_batches = int(np.ceil(len(train_images) / batch_size))
    import time

    # adam optimizer
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 0.00000008
    startsize = 1000
    train_size = train_images.shape[0]
    minibatch_maxsize_targetiter = 500
    minibatch_maxsize = train_size
    minibatch_startsize = minibatch_startsize = 1000
    b = (minibatch_maxsize_targetiter - 1) / np.log2(
        minibatch_maxsize / minibatch_startsize)  # divisor used in updating batch size

    flattened_grad, un_flatten, flattened_params = flatten_func(one_forwardpass_and_one_backward_pass, init_params)

    m = np.zeros(len(flattened_params))
    v = np.zeros(len(flattened_params))
    t = 0
    print('Results from adam optimizer with the same back-propagation as KFAC')
    print("   Iterations  |    Minibatch size  |    Train accuracy  |    Test accuracy ")
    for i in range(num_epochs):
        for j in range(num_batches):
            num_iter = i * num_batches + j
            t += 1
            # train_inputs , train_targets = batch_data(j)
            num_iter = i * num_batches + j
            minibatch_size = next_batch_size(startsize, train_size, b, num_iter)
            train_inputs, train_targets = batch_data_inc(minibatch_size, train_images, train_labels)
            if num_iter % 100 == 0:
                train_acc = accuracy(un_flatten(flattened_params), train_images, train_labels)
                test_acc = accuracy(un_flatten(flattened_params), test_images, test_labels)
                print("{:15}|{:20}|{:20}|{:20}".format(num_iter, minibatch_size, train_acc, test_acc))

            batch_grad = flattened_grad(flattened_params, train_inputs, train_targets , numlayers, layer_sizes, layer_types) # here j represents the jth batch, batch_grad is also flattened
            batch_grad = batch_grad/batch_size
            m = beta1 * m + (1 - beta1) * batch_grad
            v = beta2 * v + (1 - beta2) * (batch_grad ** 2)
            m2 = m / (1 - beta1**t)
            v2 = v / (1 - beta2**t)
            flattened_params = flattened_params - step_size * m2 / (np.sqrt(v2) + epsilon)

    trained_params = un_flatten(flattened_params)
