import numpy as np
import scipy.io as sio
import minimize

def flattenMatrix(m):
    m = m.flatten(1)
    m = m.reshape(len(m), 1)
    return m

def softmaxCost(theta, numClasses, inputSize, lambda_, data, labels):
    '''numClasses - the number of classes 
       inputSize - the size N of the input vector
       lambda_ - weight decay parameter
       data - the N x M input matrix, where each column data(:, i) corresponds to
              a single test set
       labels - an numClasses x M matrix containing the labels corresponding for the input data
       Returns: Cost and the Gradient
    '''
    theta = theta.reshape(numClasses, inputSize, order='F')
    numCases = data.shape[1]
    cost = 0
    thetagrad = np.zeros((numClasses, inputSize))
    full_prediction = np.dot(theta, data)
    max_prediction = np.max(full_prediction, 0)
    shrunk_prediction = full_prediction - max_prediction
    exp_shrunk_prediction = np.exp(shrunk_prediction)
    term = np.sum(exp_shrunk_prediction, 0)
    prediction = (exp_shrunk_prediction/term)
    
    log_term = np.log(prediction)
    cost = -1 * np.sum(labels*log_term)/numCases
    cost += ((0.5 * lambda_) * np.sum(theta**2))
   
    gp = labels - prediction
    thetagrad = (np.dot(data, gp.T)).T * (-1.0 / numCases)
    thetagrad = thetagrad + (lambda_ * theta);

    grad = flattenMatrix(thetagrad)

    return cost, grad

def softmaxPredict(theta, data):
    h1 = np.dot(theta, data)
    pred = np.argmax(h1, 0) + 1
    return pred

def softmaxTrain(inputSize, numClasses, lambda_, inputData, labels, n_iterations=100):
    theta = 0.005 * np.random.randn(numClasses * inputSize, 1);
    softmaxCostF = lambda p: softmaxCost(p, numClasses, inputSize, lambda_, inputData, labels) 
    optTheta, cost, iteration = minimize.minimize(softmaxCostF, theta, n_iterations)
    optTheta = optTheta.reshape(numClasses, inputSize, order='F')
    return optTheta
    
    
          
    
    
