"""SimpleApp.py"""

import sys

from pyspark.sql import SparkSession
from pyspark.context import SparkContext

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
        print("Usage: sgd <data_path> <num_iterations> <num_workers>", file=sys.stderr)
        sys.exit(-1)

    # Get arguments
    data_path = sys.argv[1]
    n_iterations = int(sys.argv[3])
    n_workers = int(sys.argv[4])

    # Create Spark environment
    spark = SparkSession.builder.appName("PySparkSGD").getOrCreate()

    #sc = SparkContext(sys.argv[1], "PySparkSGD")
    sc = spark.SparkContext

    #load data
    data_train, labels_train = data.load_data(sc)

    # init weight vectors
    w = [0] * len(data_train[0]) 
    num_examples = data_train.shape[0]
    lambda_ = 0.001

    #full data
    training = zip(data_train,labels_train)
    
    # training
    for i in range(n_iterations):
    	#passer un zip de data et labels pour le passer dans parallelize
        sgd = sc.parallelize(training, numSlices=n_workers) \
        .mapPartitions(lambda (x,y): helpers.calculate_stochastic_gradient(y, x, w, lambda_, num_examples)) \
        .reduce(lambda x, y: merge(x, y)).collect() \
        w = helpers.avg_model(sgd, slices) # averaging weight vector => iterative parameter mixtures
        print "Iteration %d:" % (i + 1)
        print "Model: "
        print w
        print ""    