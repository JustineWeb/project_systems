"""
helper functions to load the data
"""
from pyspark.sql import SparkSession

data_train = "/data/datasets/lyrl2004_vectors_train.dat"
data_test_0 = "/data/datasets/lyrl2004_vectors_test_pt0.dat"
data_test_1 = "/data/datasets/lyrl2004_vectors_test_pt1.dat"
data_test_2 = "/data/datasets/lyrl2004_vectors_test_pt2.dat"
data_test_3 = "/data/datasets/lyrl2004_vectors_test_pt3.dat"

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


def load_data(sc,data_ = data_train):
	'''
	Function to load the data (we are not using spark here but we could later on during the project)
	sc : spark context
	'''
	rdd = sc.textFile(data_).map(line => line.strip("")).map(line => line.split(' '))
	labels = rdd.map(line => int(line[0]))
	data = rdd.map(line => generate_dictionary(line[2:]))

	return data, labels

