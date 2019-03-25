"""SimpleApp.py"""

import sys

#from pyspark.sql import SparkSession

from pyspark.context import SparkContext

import numpy as np

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
        print("Usage: PythonSparkSGD <master> <data_path> <num_iterations> <num_workers>", file=sys.stderr)
        sys.exit(-1)

    sc = SparkContext(sys.argv[1], "PySparkSGD")

    path = sys.argv[1]
    iterations = sys.argv[3]
    workers = sys.argv[4]

    #val rdd = ... ?
    rdd = sc.textFile(path)

    