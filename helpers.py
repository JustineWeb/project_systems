"""
Helper functions to run the SGD algorithm.
"""

import random

def hinge_loss(y,x,w):
    '''
    Compute the value of the hinge loss
    x: sparse_vector
    y: label
    w: weigths vector
    '''
    val = 1 - y * x.dot(w)
    if  val < 0:
        return 0 
    elif val <= float('Inf'):
        return val
    else:
        return float('Inf')

def dot(x,y):
    """
    vanilla python dot product
    """
    dot = 0 
    for i in range(len(x)):
        dot += x[i]*y[i]
    return dot

def calculate_primal_objective(y,x,w,lambda_):
    """ 
    compute the full cost (the primal objective), that is loss plus regularizer.
    """
    v = hinge_loss(y, X, w)
    return sum(v) + lambda_ / 2 * sum(w ** 2)

def accuracy(iterator):
    """
    compute the accuracy of the model (number of well predicted points)
    """
    score = 0
    for x in iterator:
        for elem in x:
            if elem[0] == elem[1]:
                score +=1
    yield score

def prediction(iterator,w):
    """
    do the prediction using the model.
    """
    pred = []
    for x in iterator:
        temp = (x[0].dot(w) > 0) * 2 - 1
        y = x[1]
        pred.append((temp,y))
    yield pred

def batch_iter(x,y,batch_size):
    """
    create batch of the desired size.
    """
    y_batch = []
    x_batch = []
    indices = random.sample(range(len(x)),batch_size)
    for indice in indices:
        x_batch.append(x[indice])
        y_batch.append(y[indice])
    return x_batch, y_batch


def calculate_stochastic_gradient(x_n,y_n, w, lambda_, num_examples):
    """compute the stochastic gradient of loss plus regularizer.
    w: shape = (num_features)
    num_examples: N
    """

    def is_support(y_n, x_n, w):
        """a datapoint is support if max{} is not 0. """
        return y_n * x_n.dot(w) < 1

    grad = [-1*x*y_n for x in x_n] if is_support(y_n, x_n, w) else [0] * len(x_n)
    l = [w_n*lambda_ for w_n in w]
    grad = [num_examples * x + y for x,y in zip(grad,l)]
    return grad

def train(iterator,w):
    """
    training function that computes the gradient
    """
    weights = [0] * len(w)
    for x in iterator:
        weights = [w + t for w,t in zip(weights,calculate_stochastic_gradient(x[0],x[1],w,lambda_,1))] 
    yield weights
