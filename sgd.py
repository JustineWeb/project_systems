import pyspark
import sys

from pyspark.sql import SparkSession
from pyspark.context import SparkContext , SparkConf
import random
import time



#####################################################

"""
helper functions
"""
data_train = "/data/datasets/lyrl2004_vectors_train.dat"
data_test_0 = "/data/datasets/lyrl2004_vectors_test_pt0.dat"
data_test_1 = "/data/datasets/lyrl2004_vectors_test_pt1.dat"
topic_files = "/data/datasets/rcv1-v2.topics.qrels"

def generate_dictionary(datapoint):
    ''' 
    Parses and generates a dictionary from one sparse datapoint. 
    From Hogwild python implementation
    '''
    d = {0: 1.0} # Adding the bias
    for elem in datapoint:
        elem = elem.split(':')
        d[int(elem[0])] = float(elem[1])
    return d

def load_data(sc,data_ = data_train, topics_path=topic_files, selected_cat='CCAT'):
    '''
    Function to load the data (we are not using spark here but we could later on during the project)
    sc : spark context
    '''
    rdd = sc.textFile(data_).map(lambda line: line.strip("")).map(lambda line: line.split(' '))
    labels = rdd.map(lambda line: int(line[0]))
    data = rdd.map(lambda line: generate_dictionary(line[2:]))
    labels = labels.collect()

    cat = get_category_dict(topics_path)
    labels = [1 if selected_cat in cat[label] else -1 for label in labels]

    return data, labels

def get_category_dict(topics_path):
    ''' Generates the category dictionary using the topics file from:
    http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm
    From Hogwild python implementation
    '''
    categories = {}
    with open(topics_path) as f:
        content = f.readlines()
        content = [line.strip() for line in content]
        content = [line.split(' ') for line in content]
        for line in content:
            id = int(line[1])
            cat = line[0]
            if id not in categories:
                categories[id] = [cat]
            else:
                categories[id].append(cat)
    return categories

def dotproduct(x, w):
    ''' Calculates the dotproduct for sparse x and w. '''
    return sum([v * w.get(k, 0) for k, v in x.items()])

def sign(x):
    ''' Sign function '''
    return 1 if x > 0 else -1 if x < 0 else 0

def accuracy(iterator):
    score = 0
    for x in iterator:
        for elem in x:
            if elem[0] == elem[1]:
                score +=1
    yield score

def hinge_loss(y,x,w):
    '''
    Compute the value of the hinge loss
    x: sparse_vector
    y: label
    w: weigths vector
    '''
    return min(max(1 - y * dotproduct(x,w),0),float('inf'))

def calculate_primal_objective(y,x,w,lambda_,batch_size):
    '''
    compute the full cost (the primal objective), that is loss plus regularizer.
    '''
    v = hinge_loss(y, x, w)
    return v/batch_size + lambda_ / 2 * sum([(weight**2)/batch_size for weight in w.values()])

def calculate_stochastic_gradient(x_n,y_n, w, lambda_):
    '''
    compute the stochastic gradient of loss plus regularizer.
    w: shape = (num_features)
    num_examples: N
    '''
    grad = {}
    
    def is_support(y_n, x_n, w):
        """a datapoint is support if max{} is not 0. """
        return y_n * dotproduct(x_n,w) < 1
    
    supp = is_support(y_n, x_n, w)
    
    for k,v in x_n.items():
        if supp:
            grad[k] = -v*y_n
        else:
            grad[k] = 0
        
    reguralizer = {}
    for k in x_n.keys():
        reguralizer[k] = w.get(k, 0)**2 * reg
    
    for k,v in grad.items():
        grad[k] = lambda_* v + reguralizer[k]
    
    return grad

def prediction(iterator,w):
	'''
	for a given array of input and a weights vector, create a tuple containing the true value and the predicted value.
	'''
	pred = []
	for x in iterator:
		temp = (dotproduct(x[0],w) > 0) * 2 - 1
		y = x[1]
		pred.append((temp,y))
	yield pred

def train(iterator,w):
	'''
	for a given iterator and a weights vector, compute the weigths update using SGD
	'''
	weights = {}
	for x in iterator:
		grad = calculate_stochastic_gradient(x[0],x[1],w,lambda_)
		for k, v in grad.items():
			weights[k] = v + weights.get(k, 0)
	yield weights

def batch_iter(x,y,batch_size):
	'''
	Create batch of the given size given input data
	'''
	y_batch = []
	x_batch = []
	indices = random.sample(range(len(x)),batch_size)
	for indice in indices:
		x_batch.append(x[indice])
		y_batch.append(y[indice])
	return x_batch, y_batch

def compute_loss(iterator,w,lambda_,batch_size):
	'''
	compute the loss for a given iterator
	'''
	loss = 0
	for x in iterator:
		loss += calculate_primal_objective(x[1],x[0],w,lambda_,batch_size)
	yield loss 


if __name__ == "__main__":

	# create the spark context
	conf = SparkConf().setAppName("PySpark App")
	sc = SparkContext(conf = conf)
	sc.setLogLevel("Error")

	# load the data
	print('loading data')
	data_train, labels_train = load_data(sc,data_= data_train)
	data_train = data_train.collect()

	# settings
	w = {}
	num_examples = 1
	n_workers = 10
	batch_per_worker = 100
	batch_size = batch_per_worker * n_workers
	lambda_ = 0.3 /batch_size
	reg = 0.000001
	n_iterations = 1000

	# apply SGD for the given number of iteratiors
	total_time = 0
	for i in range(n_iterations):

		print("######################################################################")
		print('we are at the iteration : {}'.format(i))

		# computing the gradient step
		start = time.time()
		x_batch, y_batch = batch_iter(data_train,labels_train,batch_size= batch_size)
	    
		sgd = sc.parallelize(zip(x_batch,y_batch),numSlices= n_workers) \
		.mapPartitions(lambda it: train(it,w)) \
		.reduce(lambda x,y:{k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)})
	    
		# update the gradient
		for k,v in sgd.items():
			w[k] = w.get(k, 0) - v
	    
		# print time the SGD iteration took
		end = time.time()
		print("time:")
		print(end - start)
		total_time += end - start
	    
		# compute the cost (in case the cost explose, we use a try catch)
		try:
			print("compute loss")
			loss = sc.parallelize(zip(x_batch,y_batch),numSlices= n_workers) \
			.mapPartitions(lambda it: compute_loss(it,w,lambda_,batch_size)) \
			.reduce(lambda x,y: x+y)
			print("the loss is:")
			print(loss)
		except Exception:
			pass

		# compute the current training accuracy of the model
		print("compute train accuracy")
		acc = sc.parallelize(zip(x_batch,y_batch), numSlices= n_workers) \
		.mapPartitions(lambda it: prediction(it,w)) \
		.mapPartitions(lambda it: accuracy(it)) \
		.reduce(lambda x,y: x+y)
		acc = float(acc)/float(batch_size)
		print('the model as achieved an accuracy of:')
		print(acc)
		print("######################################################################")
	
	# print the total time took by the algorithm and save the final weight vectors
	print("Time in total")
	print(total_time)
	with open('/data/w10.txt','w') as data:
		data.write(str(w))