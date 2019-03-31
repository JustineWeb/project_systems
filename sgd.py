"""SimpleApp.py"""

from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.mllib.linalg import SparseVector

import helpers
import data

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
    data_train, labels_train = data.load_data(sc)
    data_train = data_train.collect()

    #compute size of train vectors, m
    m = 0
    for i in range(len(data_train)):
        if m < max(data_train[i],key=int):
        m = max(data_train[i],key=int)
    print(m)

    # init weight vectors and data_train vectors from the dictionnary
    data_train = [SparseVector(m+1,d) for d in data_train]
    w = [0] * len(m+1) 
    num_examples =  len(data_train)
    lambda_ = 0.001
    
    # training
    batch_size = 200
    mini_batch_size = 10
    for i in range(n_iterations):
        for j in range(batch_size):
            x_batch, y_batch = helpers.batch_iter(data_train,labels_train,batch_size= mini_batch_size)
            sgd = sc.parallelize(zip(x_batch,y_batch),numSlices= n_workers) \
            .mapPartitions(lambda it: helpers.train(it,w)) \
            .coalesce(n_workers)
        sgd = sgd.reduce(lambda x,y: [a+b for a,b in zip(x,y)])
        sgd = [elem/mini_batch_size for elem in sgd]
        w = [x + y/batch_size for x,y in zip(w,sgd)]
        print("Iteration %d:" % (i + 1))
        print("Model: ")
        print(w)
        print ("")

    """
    1) validation score to compute ( can do it at all iteration) 
    2) timer
    3) test
    """  