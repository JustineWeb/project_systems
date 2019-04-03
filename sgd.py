"""SimpleApp.py"""

from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.mllib.linalg import SparseVector

import helpers
import data
import random
import time

#logFile = "YOUR_SPARK_HOME/README.md"  # Should be some file on your system
#spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
#logData = spark.read.text(logFile).cache()

#numAs = logData.filter(logData.value.contains('a')).count()
#numBs = logData.filter(logData.value.contains('b')).count()

#print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

#spark.stop()

# check : https://gist.github.com/MLnick/4707012

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: sgd <num_iterations> <num_workers>", file=sys.stderr)
        sys.exit(-1)

    # Get arguments
    data_path = sys.argv[1]
    n_iterations = int(sys.argv[3])
    n_workers = int(sys.argv[4])

    # Create Spark environment
    #spark = SparkSession.builder.appName("PySparkSGD").getOrCreate()

    sc = SparkContext()

    #load data
    data_train, labels_train = load_data(sc)
    data_test_0, labels_test_0 = load_data(sc, data_= data_test_0)
    data_test_1, labels_test_1 = load_data(sc,data_= data_test_1)
    
    data_train = data_train.collect()
    data_test_0 = data_test_0.collect()
    data_test_1 = data_test_1.collect()

    #compute size of train vectors, m
    m = 0
    for i in range(len(data_train)):
        if m < max(data_train[i],key=int):
        m = max(data_train[i],key=int)
    print(m)

    # init weight vectors and data_train vectors from the dictionnary
    data_train = [SparseVector(m+1,d) for d in data_train]
    data_test_0 = [SparseVector(m+1,d) for d in data_test_0]
    data_test_1 = [SparseVector(m+1,d) for d in data_test_1]
    
    data_test = data_test_0 + data_test_1
    labels_test = labels_test_0 + labels_test_1
    
    w = [0] * (m+1) 
    num_examples = len(data_train)
    n_workers = 5
    lambda_ = 3/n_workers
    
    # training
    batch_size = 100 * n_workers
    total_time = 0
    for i in range(n_iterations):
        print('we are at the iteration : {}'.format(i))

        # computing the gradient
        start = time.time()
        x_batch, y_batch = batch_iter(data_train,labels_train,batch_size= batch_size)
        sgd = sc.parallelize(zip(x_batch,y_batch),numSlices= n_workers) \
        .mapPartitions(lambda it: train(it,w)) \
        .reduce(lambda x,y: [a+b for a,b in zip(x,y)])
        w = [x + y/100 for x,y in zip(w,sgd)]
        print(len(w) - w.count(0))
        end = time.time()
        print(end - start)
        total_time += end - start
        
        print("compute train accuracy")
        # computing the train accuracy
        acc = sc.parallelize(zip(x_batch,y_batch), numSlices= n_workers) \
        .mapPartitions(lambda it: prediction(it,w)) \
        .mapPartitions(lambda it: accuracy(it)) \
        .reduce(lambda x,y: x+y)
        acc = acc/len(y_batch)
        print('the model as achieved an accuracy of:')
        print(acc)


        print('computing test accuracy')
        # computing the test accuracy
        test_x_batch, test_y_batch = batch_iter(data_test,labels_test,batch_size= batch_size)
        acc = sc.parallelize(zip(test_x_batch,test_y_batch), numSlices= n_workers) \
        .mapPartitions(lambda it: prediction(it,w)) \
        .mapPartitions(lambda it: accuracy(it)) \
        .reduce(lambda x,y: x+y)
        acc = acc/len(test_y_batch)
        print('the model as achieved an accuracy of:')
        print(acc)