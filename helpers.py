import numpy as np

"""
Helper functions to run the SGD algorithm.
"""
def hinge_loss(y,X,w):
	return np.clip(1 - y * (X @ w), 0, np.inf)

def calculate_primal_objective(y,X,w,lambda_):
		"""
		compute the full cost (the primal objective), that is loss plus regularizer.
		"""
	v = hinge_loss(y, X, w)
    return np.sum(v) + lambda_ / 2 * np.sum(w ** 2)

def accuracy(y1, y2):
    return np.mean(y1 == y2)

def prediction(X, w):
    return (X @ w > 0) * 2 - 1

def calculate_accuracy(y, X, w):
    """compute the training accuracy on the training set (can be called for test set as well).
    """
    predicted_y = prediction(X, w)
    return accuracy(predicted_y, y)

def calculate_stochastic_gradient(y, X, w, lambda_, n, num_examples):
    """compute the stochastic gradient of loss plus regularizer.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    n: the index of the (one) datapoint we have sampled
    num_examples: N
    """
   
    def is_support(y_n, x_n, w):
        """a datapoint is support if max{} is not 0. """
        return y_n * x_n @ w < 1
    
    x_n, y_n = X[n], y[n]
    grad = - y_n * x_n.T if is_support(y_n, x_n, w) else np.zeros_like(x_n.T)
    grad = num_examples * np.squeeze(grad) + lambda_ * w
    return grad