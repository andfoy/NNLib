import numpy as np
import scipy
import scipy.io as sio
import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

randomness_source = np.random.rand(1, 7000000)
report_calls_to_sample_bernoulli = False
plt.ion() 

def flattenMatrix(mat):
    mat = mat.flatten(1)
    vect = mat.reshape(len(mat), 1)
    return vect

def clear():
    os.system('clear')

def argmax_over_rows(mat):
    return mat.argmax(0)

def a4_rand(requested_size, seed):
    start_i = np.mod(np.round(seed), np.round(randomness_source.shape[1] / 10)) + 1
    if start_i + np.prod(requested_size) >= randomness_source.shape[1] + 1:
       raise Exception("a4_rand failed to generate an array of that size (too big)")
    rand = randomness_source[0, start_i : start_i+np.prod(requested_size)]
    ret = rand.reshape(requested_size, order='F')
    return ret

def cd1(rbm_w, visible_data):
    '''<rbm_w> is a matrix of size <number of hidden units> by <number of visible units> 
       <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
       The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.'''
    visible_data = sample_bernoulli(visible_data)
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_data)
    binaryhid = sample_bernoulli(hidden_probability)
    g1 = configuration_goodness_gradient(visible_data, binaryhid)
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, binaryhid)
    binaryvis = sample_bernoulli(visible_probability)
    hiddenprob = visible_state_to_hidden_probabilities(rbm_w, binaryvis)
    #binaryhid2 = sample_bernoulli(hiddenprob)
    g2 = configuration_goodness_gradient(binaryvis, hiddenprob)
    ret = g1 - g2
    ret = ret.T
    return ret

def configuration_goodness(rbm_w, visible_state, hidden_state):
    '''<rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
       <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
       <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
       This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.'''
       
    G = np.mean(np.sum(np.dot(rbm_w.T, hidden_state)*visible_state, 0));
    return G

def configuration_goodness_gradient(visible_state, hidden_state):
    '''<visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
       <hidden_state> is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that   we're handling in parallel>.
       You don't need the model parameters for this computation.
       This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters. Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function. Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).'''

    d_G_by_rbm_w = np.dot(visible_state, hidden_state.T)/(np.float32(visible_state.shape[1]));
    return d_G_by_rbm_w

def extract_mini_batch(data, start_i, n_cases):
    mini_batch = data[:, start_i : start_i + n_cases]
    return mini_batch

def hidden_state_to_visible_probabilities(rbm_w, hidden_state):
    '''<rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
       <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
       The returned value is a matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
       This takes in the (binary) states of the hidden units, and returns the activation probabilities of the visible units, conditional on those states. '''
    
    term = np.dot(rbm_w.T, hidden_state)
    visible_probability = logistic(term);
    return visible_probability

def logistic(z):
    return (1/(1+np.exp(-z)))

def optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations, displ=True, model=False, rbm_w = 0):
    '''
       This trains a model that's defined by a single matrix of weights.
       <model_shape> is the shape of the array of weights.
       <gradient_function> is a function that takes parameters <model> and <data> 
       and returns the gradient (or approximate gradient in the case of CD-1) of 
       the function that we're maximizing. Note the contrast with the loss function 
       that we saw in PA3, which we were minimizing. The returned gradient is an 
       array of the same shape as the provided <model> parameter.
       This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
       This returns the matrix of weights of the trained model.
    '''
    if not model:
       model = (a4_rand(model_shape, np.prod(model_shape)) * 2 - 1) * 0.1;
    else:
       model = rbm_w
    momentum_speed = np.zeros(model_shape)
    mini_batch_size = 100;
    start_of_next_mini_batch = 0;
    for iteration_number in range(0, n_iterations+1):
        clear()
        if displ:
           displayData(model)
           plt.draw()
        print 'Iteration %d | Batch # %d\n' % (iteration_number, start_of_next_mini_batch);
        mini_batch = extract_mini_batch(training_data, start_of_next_mini_batch, mini_batch_size);
        start_of_next_mini_batch = np.mod(start_of_next_mini_batch + mini_batch_size, training_data.shape[1]);
        gradient = gradient_function(model, mini_batch)
        momentum_speed = 0.9 * momentum_speed + gradient
        model = model + momentum_speed * learning_rate
    return model

def sample_bernoulli(probabilities):
    if report_calls_to_sample_bernoulli:
       print "sample_bernoulli() was called with a matrix of size %d by %d.\n\n" % (probabilities.shape[0], probabilities.shape[1])
    seed = np.sum(flattenMatrix(probabilities))
    binary = 0 + (probabilities > a4_rand(probabilities.shape, seed))
    return binary

def visible_state_to_hidden_probabilities(rbm_w, visible_state):
    '''<rbm_w> is a matrix of size <number of hidden units> by <number of visible units>

       <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.

       The returned value is a matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.

       This takes in the (binary) states of the visible units, and returns the activation probabilities of the hidden units conditional on those states.'''
    term = np.dot(rbm_w, visible_state)
    hidden_probability = logistic(term)
    return hidden_probability

def displayData(X, example_width=False):
    '''DISPLAYDATA Display 2D data in a nice grid
       [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
       stored in X in a nice grid. It returns the figure handle h and the 
       displayed array if requested.
    '''
    if not example_width:
       example_width = np.round(np.sqrt(X.shape[1]))
    m,n = X.shape
    example_height = (n / example_width)
    # Compute number of items to display 
    display_rows = np.floor(np.sqrt(m))
    display_cols = np.ceil(m / display_rows)
    # Between images padding
    pad = 1;
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                      pad + display_cols * (example_width + pad)))
    curr_ex = 0;
    for i in np.arange(1, display_rows+1):
        for j in np.arange(1, display_cols+1):
            if curr_ex > m-1: 
               break 
            max_val = np.max(np.abs(X[curr_ex, :]))
            ly = np.int32(pad + (j - 1)  * (example_height + pad) + np.arange(0,example_height+1))
            lx = np.int32(pad + (i - 1)  * (example_width + pad) + np.arange(0,example_width+1))
            y = slice(ly[0], ly[-1])
            x = slice(lx[0], lx[-1])
            display_array[x, y] = X[curr_ex, :].reshape(example_height, example_width, order='F') / max_val;
            curr_ex += 1
        if curr_ex > m-1:
           break
    
                
    plt.imshow(display_array, cmap = plt.get_cmap('gray'))
    #plt.show()

def train_rbm(n_hid, lr_rbm, n_iterations, rbm_w = False, n_iter = False):
    clear()
    inputs = sio.loadmat('inputs.mat')['inputs']
    fcd1 = lambda rbm_w, data: cd1(rbm_w, data)
    if not rbm_w:
       rbm_w = optimize(np.array([n_hid, 784]), fcd1, inputs, lr_rbm, n_iterations, True)
    else:
       rbm_w = sio.loadmat('rbm_w.mat')['rbm_w']
       rbm_w = optimize(np.array([n_hid, 784]), fcd1, inputs, lr_rbm, n_iter, True, True, rbm_w)
    sio.savemat('rbm_w.mat', {'rbm_w':rbm_w})
    
    
        
    



