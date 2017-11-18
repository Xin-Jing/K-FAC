"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.differential_operators import make_jvp_reversemode
from autograd.misc.flatten import flatten
from data import load_mnist

# K-FAC

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """A deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities,
       and preactivations of the last layer
       """
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def last_layer_preactivations(params, inputs):
    """A deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
        returns the preactivations of the last layer"""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    #log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    log_lik = log_lik/inputs.shape[0]
    return log_lik #+ log_prior

def next_batch_size(startsize, train_size, b, k):
    # exponentially schedule for minibatch size as described in Page 42.
    # k is the number of iterations.
    return int(min(startsize*2**(k/b), train_size))

def soft_max(x):
    # shape of x : (# of samples * dimension)
    e_x = np.exp((x.T - np.max(x, 1)).T)
    return (e_x.T / np.sum(e_x,1)).T

def log_predict(outputs,layer_types):
    # return the log_probabilities given the preactivations of the last layer
    # so far only softmax function included, will include more types of last layer functions
    if layer_types[-1] == 'softmax':
        return outputs - logsumexp(outputs, axis=1, keepdims=True)  # a_l

def objective(outputs, params, targets, layer_types):
    # outputs are the preactivations from the last layer
    # will modify is for different regularization and loss functions
    log_predictions = log_predict(outputs, layer_types)
    log_lik = np.sum(log_predictions * targets) # this is negative
    return - log_lik  # L2 shouldn't be added here, since it will be taken care of later

def softmax_sampling(predictions):
    # sample the targets of the softmax layer from probabilities predicted by the model
    c = np.cumsum(predictions, 1)
    s = npr.rand(predictions.shape[0])
    return np.diff(np.concatenate((np.zeros([1, predictions.shape[0]]).T,
                                   np.tile(s, (predictions.shape[1], 1)).T <= c),
                                  axis=1),axis=1)

def one_forwardpass_and_two_backward_pass(minibatch_size,sampleminibatch_size,inputs_minibatch, targets_minibatch, params, numlayers, layer_sizes, layer_types, weight_cost, flattened_params, L2_reg):
    # back_prop that keeps track of ai's and gi's and returns flattened gradient,  A_hom_inc, G_inc, last layer preactivations and  log likelihood
    a_inc = []
    a_inc.append(inputs_minibatch) # now a_inc includes the minibatch inputs
    inputs = inputs_minibatch
    # forward prop
    for i in range(numlayers-1):
        outputs = np.dot(inputs, params[i][0]) + params[i][1] # preactivations from the i-th layer: a_i
        if layer_types[i] == 'tanh':
            inputs = np.tanh(outputs) # activations of the i-th layer: a_i, will add more activations later
        a_inc.append(inputs)  # now a_inc includes the a_i_hom = [a_i, 1], i < l
    outputs = np.dot(inputs, params[numlayers-1][0]) + params[numlayers-1][1]  # preactivations from the last layer
    log_probabilities = outputs - logsumexp(outputs, axis=1, keepdims=True)
    log_likelihood = np.sum(log_probabilities * targets_minibatch)/inputs_minibatch.shape[0] #- L2_reg * l2_norm(params)

    # Backward Prop
    D_W_b = [[np.zeros([m, n]),  # matrix of gradient of weights
              np.zeros(n)]  # matrix of gradients of baises
             for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
    # Last_layer
    if layer_types[-1] == 'softmax':
        predictions = soft_max(outputs)
        g = predictions - targets_minibatch # Gradient of objective function w.r.t. outputs of the last layer
        # note that this derivative is for negative log-likelihood without L2 penelty being added
    else:
        grad_last = grad(objective) # Gradient of objective function w.r.t. outputs of the last layer
        g = grad_last(outputs, params, targets_minibatch, layer_types) # shape: (layer_sizes[l] * num of examples)
    D_W_b[-1][0] =  np.dot(a_inc[-1].T, g)# DW of the last layer ; shape: (layer_sizes[l-1], 10)
    D_W_b[-1][1] = np.sum(g,0) # Db of the last layer

    for i in range(1,numlayers):
        if layer_types[-i-1] == 'tanh':
            g = np.dot(g,params[-i][0].T) * (1 + a_inc[-i]) * (1 - a_inc[-i]) # note that a_inc[-1] is the second last layer
        D_W_b[-i-1][0] = np.dot(a_inc[-i-1].T, g)
        D_W_b[-i-1][1] = np.sum(g, 0)
    flattened_gradient = flatten(D_W_b)[0]/minibatch_size
    flattened_gradient += weight_cost * flattened_params # L2 regularization

    # additional backward pass:
    A_hom_inc= {}
    G_inc = {}
    # last layer sampling:
    predictions_sample = predictions[0:sampleminibatch_size]
    if layer_types[-1] == 'softmax':
        g = predictions_sample - softmax_sampling(predictions_sample) # g = randomly-sampled target for softmax layer  -  predictions
    G_inc[numlayers,numlayers] = np.dot(g.T, g)/sampleminibatch_size  # G_inc[l,l] = (g_l * g_l.T)/sampleminibatch_size

    for i in range(1, numlayers):
        ai_inc_sample = a_inc[-i][0:sampleminibatch_size]  # note that a_inc[-1] is activations from the second last layer
        ai_hom_inc = np.concatenate((ai_inc_sample, np.ones([ai_inc_sample.shape[0], 1])), axis=1)
        A_hom_inc[numlayers - i , numlayers - i ] = np.dot(ai_hom_inc.T, ai_hom_inc)  # A[i,i] = ai_hom_inc * ai_hom_inc.T/sampleminibatch_size
        if layer_types[-i-1] == 'tanh':
            g = np.dot(g, params[-i][0].T) * (1 + ai_inc_sample) * (1 - ai_inc_sample)
        G_inc[numlayers-i, numlayers-i] = np.dot(g.T, g)/sampleminibatch_size

    ai_inc_sample = a_inc[0][0:sampleminibatch_size]  # a_0_sample
    ai_hom_inc = np.concatenate((ai_inc_sample, np.ones([ai_inc_sample.shape[0], 1])), axis=1)  # a_0_hom_sample
    A_hom_inc[0, 0] = np.dot(ai_hom_inc.T, ai_hom_inc) / sampleminibatch_size  # A[0,0] = (a0_hom_inc * a0_hom_inc.T)/sampleminibatch_size
    # Now we have G_inc(i,i), where i = 1,2 3 , and A_hom_inc(i,i), where i = 0, 1, 2
    return flattened_gradient,  A_hom_inc, G_inc, outputs, log_likelihood

def compute_invdiF_V(unflatten, inv_A_hom_damp, inv_G_damp, flattened_gradient, numlayers):
    # compute the product of diagonal approximate inverse Fisher and the gradient
    # unflatten is a function
    # return the flattened product, which is called proposal, or preconditioned gradient
    gradient = unflatten(flattened_gradient)
    proposal = [[0,0]  for i in range(numlayers)]
    for i in range(numlayers):
        D_weights_i = np.concatenate((gradient[i][0], gradient[i][1].reshape(1, len(gradient[i][1]))))
        tmp = inv_G_damp[i+1,i+1].dot(D_weights_i.T).dot(inv_A_hom_damp[i,i])
        proposal[i][0] = tmp[:,:-1].T  # double check the dimension here
        proposal[i][1] = tmp[:, -1]
    #flattened_proposal, _ = flatten(results)
    return proposal

def product_jacobian_proposal(proposal, params, inputs_minibatch):
    # compute the product of jacobian and proposed gradients
    j_p = make_jvp_reversemode(last_layer_preactivations)
    return j_p(params, inputs_minibatch)(proposal).T  # dim(10 * 1008)

def compute_quadmodel_hyperparameters(proposal, flattened_gradient, outputs, params, inputs_minibatch, recent_lambda, weight_cost, ratio_vFv=1 ):
    # See section 6.4 and 7 from the paper
    # For now, we only compute alpha from section 6.4, and will implement the computation of both alpha and mu from section 7 later.
    # To computer the hyperparameters, we first need proposal.T * F * proposal, where F is the exact Fisher given by
    # the data from the current minibatch. To reduce computational costs, we factorize F as J.T * Fr * J, where J is
    # the Jacobian of preactivations from the last layer w.r.t all the parameters (i.e. weights and biases), and Fr
    # is the Fisher matrix of the loss function w.r.t preactivtaions from the last layer.
    # Therefore, J should be in the shape of (dim of last layer * num of parameters), and Fr should be in the shape of
    # (dim of last layer * dim of last layer). We will again factorize Fr as B * B.T, where B is any matrix that satisfies
    # Fr = B * B.T. So now the scalar we want to compute: proposal.T * F * proposal, can be written as :
    #                                   proposal.T * J.T * B * B.T * J * proposal
    # We can compute this by first computing half = B * B.T * J * proposal, then compute half.T * half.
    # Note that all "*"  here are dot product, not element-wise product.

    # For now,  we will implement Fr for the Categorical Logits Negative Log Prob Loss, and will implement other kinds
    # of loss functions later

    # For Categorical Logits Negative Log Prob Loss, the Fr with respect to the inputs(logits) is given by:
    # Fr = diag(p) - p * p.T
    # where p  = softmax(logits), and Fr can be factorized as Fr = B * B.T, where B = diag(q) - p * q.T
    # where q is the entry-wise square root of p. This factorization is based on information from the tensorflow
    # implementation of CategoricalLogitsNegativeLogProbLoss in kfac/python/ops/loss_functions.py

    if ratio_vFv != 1:
        # sample a fraction of outputs accroding to the ratio.
        minibatch_size = int(ratio_vFv * outputs.shape[0])
        inputs_minibatch = inputs_minibatch[0: minibatch_size]
        outputs = outputs[0: minibatch_size]
    else:
        minibatch_size = outputs.shape[0]

    p = soft_max(outputs).T # last layer size * minibatch_size
    q = np.sqrt(p)
    # now compute jacobian-proposal product
    j_p = product_jacobian_proposal(proposal, params, inputs_minibatch) # last layer size * minibatch_size

    # not working yet
    # half =  (q * j_p  - q.dot(p.T).dot(j_p))/minibatch_size # last layer size * minibatch_size
    # pFp = np.dot(half.T,half) # minbatch_size * minibatch_size, symmetric
    # pFp = np.sum(pFp)

    # using the following as a replacement, but how to improve this ??
    pFp = 0
    for i in range(minibatch_size):
        half = q[:, i] * j_p[:, i] - np.outer(q[:, i], p[:, i]).dot(j_p[:, i])
        pFp += np.dot(half.T,half)
    pFp = pFp/minibatch_size

    flattened_proposal, _ = flatten(proposal)
    m11 = pFp + (recent_lambda + weight_cost)*np.dot(flattened_proposal, flattened_proposal)
    # will implement m12, m22,later for calculation of alpha and mu in section 7
    c = np.dot(flattened_gradient, flattened_proposal)
    alpha = -(c/m11)
    update = alpha * flattened_proposal
    quad_model_change = 0.5 * alpha * c # quad_model_change = qmodel(alpha*precon_grad) - qmodel(0)
    return update, quad_model_change


def KFAC(num_iter, init_params, initlambda, layer_sizes,layer_types, train_inputs, train_targets, testing_inputs, testing_targets, train_with_increasing_batch_size, L2_reg = 1):

    train_size = train_inputs.shape[0]
    numlayers = len(layer_sizes) - 1

    # parameters for lambda, see initSection 6.5
    recent_lambda = initlambda
    # initlambda = 1  a value usually more appropriate for classification nets
    lambda_drop = 19 / 20
    lambda_boost = 1 / lambda_drop
    # drop and boost are used to increase and decrease lambda when lambda is being adjusted
    lambda_max = float('Inf')
    lambda_min = 0
    # lambda_min = 1e-3
    T1 = 5 # update lambda every T1 iterations

    weight_cost = 1e-5  # standard L_2 weight-decay

    # parameters for Gamma, see section 6.6
    gamma = np.sqrt(initlambda + weight_cost)  # initialize gamma, it is then adjusted using a different rule
    gamma_drop = np.sqrt(lambda_drop)
    gamma_max = 1  # This is arbitrarily set, just to prevent gamma from stucking at high values.
    gamma_min = np.sqrt(weight_cost)
    T2 = 20  # update gamma every T2 iterations
    w2 = gamma_drop**T2

    # an exponentially increasing schedule for minibatch size. See section 13.
    minibatch_startsize = 1000
    minibatch_maxsize = train_size  # for other neural network optimization problems, small minibatch_maxsize may be appropriate
    minibatch_maxsize_targetiter = 500  # there is a lot of room for tuning here.  Really the mini-batch size should probably be adapted intelligently somehow (e.g. as in the Byrd et al. paper)
    b = (minibatch_maxsize_targetiter - 1) / np.log2(minibatch_maxsize / minibatch_startsize)  # divisor used in updating batch size

    ratio_sample = 1 / 8 # Fraction of data from mini-batch to use in estimating 2nd-order statistics used in Fisher approximation

    # Fraction of data from mini-batch to use in computing matrix-vector products with exact Fisher (when deterministic learning rate and momentum decay constant)
    ratio_vFv = 1 # using a value below 1 is dangerous and should be used with caution.

    T3 = 20  # update approximate Fisher inverse every T3 iterations
    params = init_params
    flattened_params, unflatten = flatten(params)

    A_hom = {} # a dictionary to store statistics a_hom_i*a_hom_j.T. See sections 3 to 5.
    G = {} # a dictionary to store statistics g_i*g_j.T
    A_hom_damp = {}  #  dictionaries to store damped statistics
    G_damp = {}
    inv_A_hom_damp = {} #  dictionaries to store inverse of damped statistics
    inv_G_damp = {}
    maxfade_stats = 0.95  # The rate of exponential decay used for estimating the E(A_hom) and E(G)
    for i in range(numlayers):
        A_hom[i,i] = np.zeros([layer_sizes[i]+1,layer_sizes[i]+1])
        G[i+1,i+1] = np.zeros([layer_sizes[i+1],layer_sizes[i+1]])
        A_hom_damp[i,i] = np.zeros([layer_sizes[i]+1,layer_sizes[i]+1])
        G_damp[i+1,i+1] = np.zeros([layer_sizes[i+1],layer_sizes[i+1]])
        inv_A_hom_damp[i,i] = np.zeros([layer_sizes[i]+1,layer_sizes[i]+1])
        inv_G_damp[i+1,i+1] = np.zeros([layer_sizes[i+1],layer_sizes[i+1]])

    for iter in range(num_iter):
        if train_with_increasing_batch_size:
            minibatch_size = next_batch_size(minibatch_startsize,train_size, b, iter)
        else:
            minibatch_size = 256
        sampleminibatch_size = int(ratio_sample * minibatch_size)
        idx = np.random.randint(train_size, size= minibatch_size)
        inputs_minibatch = train_inputs[idx,:]  # shape :(minibatch_size, dim of input)
        targets_minibatch = train_targets[idx,:] # Now we have the minibatch used for this iteration
        # Now we need to perform one forward and two backward pass to estimate the gradient. Please see sections 3 to 5
        # A_hom, and G.  It also returns outputs, which are preactivations of the last layer.This will be needed later in calculating hyperparams from the quadratic model.
        flattened_gradient, A_hom_inc, G_inc , outputs, log_likelihood= one_forwardpass_and_two_backward_pass(minibatch_size,
                                                                                                              sampleminibatch_size,inputs_minibatch, targets_minibatch,
                                                                                                              params, numlayers, layer_sizes, layer_types,
                                                                                                              weight_cost, flattened_params,
                                                                                                              L2_reg)

        # now update A_hom and G
        fade = min( 1 - 1/(iter+1), maxfade_stats )
        fadein = 1-fade

        for i in range(numlayers):
            A_hom[i, i] = fade * A_hom[i,i] + fadein * A_hom_inc[i,i]
            G[i+1,i+1] = fade * G[i+1,i+1] + fadein * G_inc[i+1, i+1]
            A_hom[i, i] = (A_hom[i, i] + A_hom[i, i].T)/2
            G[i + 1, i + 1] = (G[i + 1, i + 1]+ G[i + 1, i + 1].T)/2

        # Now check whether we need to update gamma, see section 6.6
        if np.mod(iter,T2) == 0:
            gammas = [max(w2*gamma, gamma_min), gamma, min((1/w2) * gamma, gamma_max)]
        else:
            gammas = [gamma]

        # Now Check whether we need to recompute the approximate Fisher inverse
        if np.mod(iter, T3) == 0 or iter <= 3:
            refreshInvApprox = True
        else:
            refreshInvApprox = False

        lst = [[],[],[]] # a list to store possible updates, quadratic model change, and gammas proposed
        for each_gamma in gammas:
            if refreshInvApprox:
                for i in range(numlayers):
                    pi = np.sqrt((np.trace(A_hom[i,i])/(layer_sizes[i]+1))/(np.trace(G[i+1,i+1])/layer_sizes[i+1]))
                    # now compute damped A_hom and G, and the inverse of them
                    A_hom_damp[i,i] = A_hom[i,i] + pi * each_gamma * np.eye(layer_sizes[i]+1)
                    G_damp[i+1,i+1] =  G[i+1,i+1] + (1/pi) * each_gamma * np.eye(layer_sizes[i+1])
                    inv_A_hom_damp[i,i] = np.linalg.inv(A_hom_damp[i,i])
                    inv_G_damp[i+1, i+1] = np.linalg.inv(G_damp[i+1,i+1])
            # Now compute the update proposal, flattened
            update_proposal = compute_invdiF_V(unflatten, inv_A_hom_damp, inv_G_damp, flattened_gradient, numlayers)
            # Now compute the final update given by alpha * propsal
            possible_update, quad_model_change = compute_quadmodel_hyperparameters(update_proposal, flattened_gradient, outputs, params, inputs_minibatch,recent_lambda, weight_cost, ratio_vFv=1)
            lst[0].append(possible_update)
            lst[1].append(quad_model_change)
            lst[2].append(each_gamma)

        # the index of the lowest quad_model_change
        min_index = np.argmin(lst[1])
        min_quad_model_change = lst[1][min_index]
        gamma = lst[2][min_index]
        final_update = lst[0][min_index]
        flattened_params += final_update
        params = unflatten(flattened_params)

        if iter % T1 == 0:
            # adjust lambda according to Levenberg-Marquart style adjustment rule. Please see section 6.5
            denominator = -min_quad_model_change
            new_log_likelihood = log_posterior(params, inputs_minibatch, targets_minibatch, L2_reg)
            rho = (new_log_likelihood - log_likelihood) / denominator

            if log_likelihood - new_log_likelihood > 0:
                rho = float('-Inf')

            if rho < 0.25:
                recent_lambda = min(recent_lambda * (lambda_boost**T1), lambda_max)
            elif rho > 0.75:
                recent_lambda = max(recent_lambda * (lambda_drop**T1), lambda_min)

        if iter % 100 == 0: # counting time and printing statistics
            if iter == 0:
                time_elapsed = 0
            else:
                end = time.time()
                time_elapsed = end - start
            start = time.time()
            train_acc = accuracy(params, train_inputs, train_targets)
            test_acc = accuracy(params, testing_inputs, testing_targets)
            print("{:15}|{:20}|{:20}|{:20}|{:20}".format(iter, minibatch_size, train_acc, test_acc, time_elapsed))
    return params

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)

if __name__ == '__main__':

    print('-----------------------------------------')

    # Model parameters
    layer_sizes = [784, 200, 100, 10]
    layer_types = ['tanh', 'tanh', 'softmax']
    L2_reg = 1.0
    param_scale = 0.1
    init_params = init_random_params(param_scale, layer_sizes)
    num_iter = 1000
    print("Loading training data...")
    N, train_images, train_labels, test_images, test_labels = load_mnist()
    train_inputs = train_images
    train_targets = train_labels
    testing_inputs = test_images
    testing_targets = test_labels
    train_with_increasing_batch_size = True # if False, then will train with fixed batch size of 256.

    print('Results from KFAC')
    print( "   Iterations  |    Minibatch size  |    Train accuracy  |    Test accuracy   |    Time elapsed during this 100 iterations   ")
    import time
    # The initial lambda value.  Will be problem dependent and should be adjusted based on behavior of first few dozen iterations.
    initlambda = 0  #  1 is more appropriate for classification nets. You may want to try initlambda = 150 for an autoencoder.

    optimized_params = KFAC(num_iter, init_params, initlambda, layer_sizes,layer_types, train_inputs, train_targets, testing_inputs, testing_targets, train_with_increasing_batch_size, L2_reg = 1)


