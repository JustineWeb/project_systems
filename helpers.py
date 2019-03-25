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

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def calculate_stochastic_gradient(y, X, w, lambda_,batch_size, num_examples):
    """compute the stochastic gradient of loss plus regularizer.
    X: the dataset matrix, shape = (num_examples, num_features)
    y: the corresponding +1 or -1 labels, shape = (num_examples)
    w: shape = (num_features)
    num_examples: N
    """
	def is_support(y_n, x_n, w):
		"""a datapoint is support if max{} is not 0. """
		return y_n * x_n @ w < 1
	
	gradient = 0

	for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
		for x,y in tx_batch, y_batch:
			grad = - y * x.T if is_support(y, x, w) else np.zeros_like(x.T)
			grad = num_examples * np.squeeze(grad) + lambda_ * w
			gradient += grad
    
    return gradient/batch_size
   
