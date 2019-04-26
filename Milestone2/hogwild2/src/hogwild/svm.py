import random
from hogwild import ingest_data
from hogwild import settings as s
from hogwild.utils import dotproduct, sign


class SVM:
    def __init__(self, learning_rate, lambda_reg, dim,w):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__dim = dim
        self.__w = w

    def __getLearningRate(self):
        return self.__learning_rate

    def __getRegLambda(self):
        return self.__lambda_reg

    def __getDim(self):
        return self.__dim

    def __getW(self):
        return self.__w

    def fit(self, data, labels):
        '''
        Calculates the gradient and train loss.
        '''
        total_delta_w = {}
        train_loss = 0
        for x, label in zip(data, labels):
            xw = dotproduct(x, self.__w)
            if self.__misclassification(xw, label):
                delta_w = self.__gradient(x, label)
            else:
                delta_w = self.__regularization_gradient(x)
                if update:
                    self.update_weights(delta_w)
            for k, v in delta_w.items():
                w[k] += self.__getLearningRate() * v
            train_loss += max(1 - label * xw, 0)
            train_loss += self.__regularizer(x)
        return train_loss / len(labels)

    def loss(self, data, labels):
        ''' Returns the MSE loss of the data with the true labels. '''
        total_loss = 0
        for x, label in zip(data, labels):
            xw = dotproduct(x, self.__w)
            total_loss += max(1 - label * xw, 0)
            total_loss += self.__regularizer(x)
        return total_loss / len(labels)

    def __regularizer(self, x):
        ''' Returns the regularization term '''
        w = self.__getW()
        return self.__getRegLambda() * sum([w[i] ** 2 for i in x.keys()]) / len(x)

    def __regularizer_g(self, x):
        '''Returns the gradient of the regularization term  '''
        w = self.__getW()
        return 2 * self.__getRegLambda() * sum([w[i] for i in x.keys()]) / len(x)

    def __gradient(self, x, label):
        ''' Returns the gradient of the loss with respect to the weights '''
        regularizer = self.__regularizer_g(x)
        return {k: (v * label - regularizer) for k, v in x.items()}

    def __regularization_gradient(self, x):
        ''' Returns the gradient of the regularization term for each datapoint '''
        regularizer = self.__regularizer_g(x)
        return {k: regularizer for k in x.keys()}

    def __misclassification(self, x_dot_w, label):
        ''' Returns true if x is misclassified. '''
        return x_dot_w * label < 1

    def predict(self, data):
        ''' Predict the labels of the input data '''
        return [sign(dotproduct(x, self.__w)) for x in data]