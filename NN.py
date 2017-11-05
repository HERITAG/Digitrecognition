#!/usr/bin/env python 

from __future__ import print_function
from Build_Training_data_1 import W2V
import sys
from itertools import cycle, izip

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report as cr

np.random.seed(42)

class MLPClassifier():
    """Multi Layered Perceptron with one hidden layer"""

    def __init__(self, n_hidden=50, learning_rate=0.1, SGD=False):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.SGD = SGD

    def softmax(self, x):
        """softmax normalization"""
        np.exp(x, x)
        x /= np.sum(x, axis=1)[:, np.newaxis]

    def fit(self, X, y, max_epochs=1):
        # uniform labels
        self.lb = LabelBinarizer()
        y = self.lb.fit_transform(y)
        #print(y)
        # get all sizes
        n_samples, n_features = X.shape
        self.n_outs = y.shape[1]
        #print(self.n_outs)
        n_iterations = int(max_epochs * n_samples)

        # initialize weights #NOTE smart initialization
        nO = np.sqrt(n_features)
        nH = np.sqrt(self.n_hidden)
        self.weights1_ = np.random.uniform(-1/nO, 1/nO, size=(n_features, self.n_hidden))
        self.bias1_ = np.zeros(self.n_hidden)
        self.weights2_ = np.random.uniform(-1/nH, 1/nH, size=(self.n_hidden, self.n_outs))
        self.bias2_ = np.zeros(self.n_outs)

        if self.SGD:
            # NOTE Stochastic Gradient Descent
            # initialize hidden-layer and output layer matrices 
            x_hidden = np.empty((1, self.n_hidden))
            delta_h = np.empty((1, self.n_hidden))
            x_output = np.empty((1, self.n_outs))
            delta_o = np.empty((1, self.n_outs))

            for it in xrange(1, max_epochs+1):
                for j in xrange(n_samples):
                    self._forward(X[j, None], x_hidden, x_output)
                    self._backward(X[j, None], y[j, None], x_hidden, x_output, delta_o, delta_h)
                pred = self.predict(X)
                #print("p:",pred)
                #print("1: ",cr(y1, pred))

        else:
            # NOTE Gradient Descent
            # initialize hidden-layer and output layer matrices 
            x_hidden = np.empty((n_samples, self.n_hidden))
            delta_h = np.empty((n_samples, self.n_hidden))
            x_output = np.empty((n_samples, self.n_outs))
            delta_o = np.empty((n_samples, self.n_outs))

            # adjust weights by a forward pass and a backward error propagation
            for i in xrange(max_epochs):
                self._forward(X, x_hidden, x_output)
                self._backward(X, y, x_hidden, x_output, delta_o, delta_h)
                pred = self.predict(X)
                print("2: ",cr(y1, pred))

    def sigmoid(self,x):
            return 1/(1+np.exp(-x/1))
        

    # predict test patterns
    def predict(self, X):
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(X, x_hidden, x_output)
        return self.lb.inverse_transform(x_output)

    def _forward(self, X, x_hidden, x_output):
        """Forward pass through the network"""
        #print("XX: ",X.shape)
        x_hidden[:] = np.dot(X, self.weights1_)
        x_hidden += self.bias1_
        x_hidden = self.sigmoid(x_hidden)
        x_output[:] = np.dot(x_hidden, self.weights2_)
        x_output += self.bias2_

        # apply softmax normalization
        self.softmax(x_output)

    def _backward(self, X, y, x_hidden, x_output, delta_o, delta_h):
        """Backward error propagation to update the weights"""

        # calculate derivative of output layer
        delta_o[:] = y - x_output
        delta_h[:] = np.dot(delta_o, self.weights2_.T) 

        # update weights
        self.weights2_ += self.learning_rate * np.dot(x_hidden.T, delta_o)
        self.bias2_ += self.learning_rate * np.mean(delta_o, axis=0)
        self.weights1_ += self.learning_rate * np.dot(X.T, delta_h)
        self.bias1_ += self.learning_rate * np.mean(delta_h, axis=0)

def collect_data(file_loc):
    lines=[]
    start = 21
    end = 52
    data = []
    output = []
    with open(file_loc, "r") as f:
        for i, line in enumerate(f.read().split('\n')):
            if i>=start and i<=end:
                lines.append([int(x) for x in line])

            if i == end+1:
                y = int(line.strip())
                lines = np.array(lines)
                X = np.array(lines).reshape(-1,).tolist()
                data.append(X)
                output.append(y)
                #print (np.array(data).shape)
                # print len(output)
                start = i
                end = i + 32
                lines = []
    #print(np.array(output))
    return np.array(data),np.array(output)

if __name__ == '__main__':
    # from sklearn import datasets
    # iris = datasets.load_iris()
    # data = iris.data
    # X1 = iris.X1
    #file_tra = '/home/muzammil/Desktop/optdigits-orig.tra'
    #file_test = '/home/muzammil/Desktop/optdigits-orig.txt'
    data_input = W2V.wordvec
    data_output = W2V.lisvec
    xtrain = data_input[:9000]
    ytrain = data_output[:9000]
    xtest =data_input[9000:]
    ytest = data_output[9000:]
    #X,y1 = collect_data(file_tra)
    #print(X.shape)
    clf = MLPClassifier(n_hidden=50, learning_rate=0.01, SGD=True)
    clf.fit(xtrain, ytrain, max_epochs=200)
    #D,O = collect_data(file_test)
    pred = clf.predict(xtest)
    #print("O: ",O)
    print("pred: ",pred)
    print("0: ",cr(ytest, pred))

