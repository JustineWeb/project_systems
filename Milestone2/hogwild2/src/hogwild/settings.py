import os

LOCAL_PATH = os.path.join((os.sep)
                          .join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[0:-2]),
                          'resources/rcv1')

TEST_FILES = [os.path.join(DATA_PATH, x) for x in []]
'''
reset to use to work
'''
N_WORKERS = 5
TRAIN_FILE = "lyrl2004_vectors_train.dat"
TOPICS_FILE = "rcv1-v2.topics.qrels"
TEST_FILES = ['lyrl2004_vectors_test_pt0.dat', 'lyrl2004_vectors_test_pt1.dat', 'lyrl2004_vectors_test_pt2.dat', 'lyrl2004_vectors_test_pt3.dat']
lock_free = True
subset_size = 100  # Number of datapoints to train on each epoch
# Learning rate for SGD. The term (100/subset_size) is used to adapt convergence to different subset sizes than 100
learning_rate = 0.03 * (100 / subset_size) / N_WORKERS
validation_split = 0.1  # Percentage of validation data
epochs = 1000  # Number of training iterations over subset on each node
persistence = 15  # Abort if after so many epochs learning rate does not decrease
lambda_reg = 1e-5  # Regularization parameter
