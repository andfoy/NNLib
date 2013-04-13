import numpy as np
import scipy.io as sio
import minimize
import rbm

"""NNLib.py

This module contains a series of functions required for training a
simple Feed-Forward Neural Network using L2 Weight Decay.

The functions were translated of Andrew's Ng mlclass (2011) ex4. 
 

"""

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def sigmoidGradient(a):
    return a*(1-a)

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, fan_in))
    numel = W.shape[0]*W.shape[1]
    W = np.sin(np.arange(1,numel+1)).reshape(W.shape[0], W.shape[1], order='F')/10.0
    return W

def checkNNGradients(lambda_=None):
    if lambda_== None:
       lambda_ = 0
    input_layer_size = 3;
    hidden_layer1_size = 5;
    hidden_layer2_size = 4;
    num_labels = 3;
    m = 5;
    Theta1 = debugInitializeWeights(hidden_layer1_size, input_layer_size);
    Theta2 = debugInitializeWeights(hidden_layer2_size, hidden_layer1_size);
    Theta3 = debugInitializeWeights(num_labels, hidden_layer2_size);
    #b1 = np.zeros((hidden_layer1_size, 1))
    #b2 = np.zeros((hidden_layer2_size, 1))
    #b3 = np.zeros((num_labels, 1))
    X  = debugInitializeWeights(input_layer_size, m)
    y = 1 + np.mod(np.arange(1, m+1), num_labels).T
    Yv = np.zeros((num_labels*m, 1))
    col = np.arange(0, m)
    Yv[(num_labels*col + y.T)-1] = 1
    Y = Yv.reshape(num_labels, m, order='F')
    nn_params = np.concatenate((Theta1.flatten(1).reshape(len(Theta1.flatten(1)), 1), 
                                Theta2.flatten(1).reshape(len(Theta2.flatten(1)), 1), 
                                Theta3.flatten(1).reshape(len(Theta3.flatten(1)), 1)))
                                #b1, b2, b3))
    costFunc = lambda p: CostFunction(p, input_layer_size, hidden_layer1_size,
                                      hidden_layer2_size, num_labels, X, Y, lambda_)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    if diff < 1e-9:
       print "The implementation is correct!, the relative difference is small (less than 1e-9). \nRelative Difference: %g\n\n" % (diff)
    else:
       print "Check Implementation\n\n"

def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    numel = theta.shape[0]*theta.shape[1]
    for p in range(0, numel):
        perturb[p] = e
        loss1 = J(theta-perturb)[0]
        loss2 = J(theta+perturb)[0]
        numgrad[p] = (loss2-loss1)/(2*e)
        perturb[p] = 0
    return numgrad

def NNCostFunction(nn_params, input_layer_size, hidden_layer1_size,
                   hidden_layer2_size, num_labels, X, Y, lambda_):

    Theta1 = nn_params[0:(hidden_layer1_size*(input_layer_size+1))].reshape(
             hidden_layer1_size,(input_layer_size+1))

    Theta2 = nn_params[(hidden_layer1_size*(input_layer_size+1)):((hidden_layer2_size*(
             hidden_layer1_size+1))+((hidden_layer1_size * (input_layer_size+1))))].reshape(
             hidden_layer2_size, (hidden_layer1_size+1))

    Theta3 = nn_params[((hidden_layer2_size*(hidden_layer1_size+1))+(
             (hidden_layer1_size*(input_layer_size + 1)))):].reshape(
             num_labels, (hidden_layer2_size+1))
    

    m = X.shape[0]
    m = np.float(m)
    Z2 = np.dot(Theta1, np.concatenate((np.ones((m, 1)), X), 1).T)
    A2 = sigmoid(Z2)
    Z3 = np.dot(Theta2, np.concatenate((np.ones((1, m)), A2)))
    A3 = sigmoid(Z3)
    Z4 = np.dot(Theta3, np.concatenate((np.ones((1, m)), A3)))
    A4 = sigmoid(Z4)
    
    #Yv = np.zeros((num_labels*m, 1))
    #col = np.arange(0, m)
    #Yv[(num_labels*col + y.T)-1] = 1
    #Y = Yv.reshape(num_labels, m, order='F')
    
    J = (1.0/m)*np.sum(-Y*np.log(A4) - (1-Y)*np.log(1-A4))
    J = J + ((lambda_/(2*m))*(np.sum((Theta1[:, 1:]**2))+
        np.sum((Theta2[:, 2:]**2))+np.sum((Theta3[:, 1]**2))))
    
    D4 = A4 - Y
    D3 = np.dot(Theta3.T, D4)*sigmoidGradient(np.concatenate(
         (sigmoid(np.ones((1, A3.shape[1]))), A3)))
    D2 = np.dot(np.concatenate((np.ones((1, Theta2.shape[1])), Theta2)).T, D3)*sigmoidGradient(
         np.concatenate((sigmoid(np.ones((1, A2.shape[1]))), A2)))

    d1 = np.dot(D2, np.concatenate((np.ones((m, 1)), X), 1))
    d2 = np.dot(D3[1:, :], np.concatenate((np.ones((1, A2.shape[1])), A2)).T)
    d3 = np.dot(D4, np.concatenate((np.ones((1, A3.shape[1])), A3)).T)

    Theta1[:, 0] = 0
    Theta2[:, 0] = 0
    Theta3[:, 0] = 0

    T1grad = (d1[1:, :]/m) + ((lambda_ / m)*Theta1)
    T2grad = (d2/m) + ((lambda_ / m)*Theta2)
    T3grad = (d3/m) + ((lambda_ / m)*Theta3)
    
    T1grad = T1grad.flatten(1)
    T2grad = T2grad.flatten(1)    
    T3grad = T3grad.flatten(1)
    
    grad = np.concatenate((T1grad.reshape(len(T1grad.flatten(1)), 1), 
                           T2grad.reshape(len(T2grad.flatten(1)), 1), 
                           T3grad.reshape(len(T3grad.flatten(1)), 1)))

    return J, grad

def CostFunction(T, inputSize, hid1Size, hid2Size, numClasses, inputs, labels, lambda_):
    T1 = T[0:(hid1Size*inputSize)].reshape(hid1Size,inputSize)
    T2 = T[(hid1Size*inputSize):(hid1Size*inputSize)+(hid2Size*hid1Size)].reshape(hid2Size,hid1Size)
    T3 = T[(hid1Size*inputSize)+(hid2Size*hid1Size):(hid1Size*inputSize)+(hid2Size*hid1Size)+(
         hid2Size*numClasses)].reshape(numClasses,hid2Size)
    #b1 = T[(hid1Size*inputSize)+(hid2Size*hid1Size)+(hid2Size*numClasses):(hid1Size*inputSize)+
         #(hid2Size*hid1Size)+(hid2Size*numClasses)+hid1Size ].reshape(hid1Size, 1)
    #b2 = T[(hid1Size*inputSize)+(hid2Size*hid1Size)+(hid2Size*numClasses)+hid1Size:(hid1Size*inputSize)+
         #(hid2Size*hid1Size)+(hid2Size*numClasses)+hid1Size+hid2Size ].reshape(hid2Size, 1)
    #b3 = T[(hid1Size*inputSize)+(hid2Size*hid1Size)+(hid2Size*numClasses)+hid1Size+hid2Size:].reshape(numClasses, 1)

    m = np.float32(inputs.shape[1])
    Z2 = np.dot(T1, inputs) #+ b1
    A2 = sigmoid(Z2)
    Z3 = np.dot(T2, A2) #+ b2
    A3 = sigmoid(Z3)
    Z4 = np.dot(T3, A3) #+ b3
    A4 = sigmoid(Z4)  
 
    J = (1.0/m)*np.sum(-labels*np.log(A4) - (1-labels)*np.log(1-A4))
    J += ((lambda_/(2*m))*(np.sum((T1**2))+np.sum((T2**2))+np.sum((T3**2))))
    
    D4 = A4 - labels
    D3 = np.dot(T3.T, D4)*sigmoidGradient(A3)
    D2 = np.dot(T2.T, D3)*sigmoidGradient(A2)
    
    d1 = np.dot(D2, inputs.T)
    d2 = np.dot(D3, A2.T)
    d3 = np.dot(D4, A3.T)
  
    T1grad = (d1/m) + ((lambda_/ m)*T1)
    T2grad = (d2/m) + ((lambda_ / m)*T2)
    T3grad = (d3/m) + ((lambda_ / m)*T3)
    
    #b1grad = np.sum(D2, 1)
    #b2grad = np.sum(D3, 1)
    #b3grad = np.sum(D4, 1)

    
    grad = np.concatenate((T1grad.reshape(len(T1grad.flatten(1)), 1), 
                           T2grad.reshape(len(T2grad.flatten(1)), 1), 
                           T3grad.reshape(len(T3grad.flatten(1)), 1)))
                           #b1grad.reshape(len(b1grad), 1), 
                           #b2grad.reshape(len(b2grad), 1), 
                           #b3grad.reshape(len(b3grad), 1)))
    
    return J, grad
    

def predict(T1, T2, T3, data):
    h1 = sigmoid(np.dot(T1, data))
    h2 = sigmoid(np.dot(T2, h1))
    h3 = sigmoid(np.dot(T3, h2))
    pred = np.argmax(h3, 0) + 1
    return pred

def trainNN(inputSize, hid1Size, hid2Size, numClasses, lambda_, inputData, labels, n_iterations=100, displ=True):
    if displ:
       sel = np.random.permutation(inputData.shape[1])
       sel = sel[0:100]
       rbm.displayData(inputData[:, sel].T)
    T1 = debugInitializeWeights(hid1Size, inputSize)
    T2 = debugInitializeWeights(hid2Size, hid1Size)
    T3 = debugInitializeWeights(numClasses, hid2Size)
    b1 = np.zeros((hid1Size, 1))
    b2 = np.zeros((hid2Size, 1))
    b3 = np.zeros((numClasses, 1))
    T = np.concatenate((T1.reshape(len(T1.flatten(1)), 1), 
                        T2.reshape(len(T2.flatten(1)), 1), 
                        T3.reshape(len(T3.flatten(1)), 1),
                        b1, b2, b3))

    NNCost = lambda p: CostFunction(p, inputSize, hid1Size, hid2Size, numClasses, inputData, labels, lambda_)
    T, cost, iteration = minimize.minimize(NNCost, T, n_iterations)

    T1 = T[0:(hid1Size*inputSize)].reshape(hid1Size,inputSize)
    T2 = T[(hid1Size*inputSize):(hid1Size*inputSize)+(hid2Size*hid1Size)].reshape(hid2Size,hid1Size)
    T3 = T[(hid1Size*inputSize)+(hid2Size*hid1Size):(hid1Size*inputSize)+(hid2Size*hid1Size)+(
         hid2Size*numClasses)].reshape(numClasses,hid2Size)

    pred = predict(T1, T2, T3, inputData)
    return pred

    
    
    
    
    
    
    
  
  
