#!/usr/bin/python2
# -*-coding:utf-8-*-
from __future__ import division

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def cal_pr(actual, predict):
    t = 0
    for i in range(actual.size):
        t += 1 if actual[i] == predict[i] else 0

    return t / actual.size


class LogisticRegressionModel:

    def __init__(self):
        self.theta = None
        self.x = None
        self.y = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, x, y, alpha=0.001, iterNums=502):
        self.x = np.c_[x, np.ones(x.shape[0])]
        self.y = y.reshape(y.size, 1)
        m, d = self.x.shape
        self.theta = np.zeros((d, 1))

        for i in range(1, iterNums):
            predict = self.sigmoid(self.x.dot(self.theta))
            diff = predict - self.y
            gradient = self.x.T.dot(diff)
            self.theta -= alpha * gradient
            if i == iterNums - 1 or i % 10 == 1:
                print "第%d次迭代,在训练数据上的表现:accuracy:%f" % (i, cal_pr(self.y, np.around(predict)))

    def predict(self, x):
        with_one = np.c_[x, np.ones(x.shape[0])]
        return self.sigmoid(with_one.dot(self.theta))


def show_data():
    mnist = input_data.read_data_sets("MNIST_data/")
    train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, \
                                       mnist.test.images, mnist.test.labels

    plt.gray()
    for i in range(train_y.size):
        plt.matshow(train_x[i].reshape(28, 28))
        print "即将展示的数字是:%d" % train_y[i]
        plt.show()


def tran2logit(y):
    return np.apply_along_axis(lambda x : 0 if x < 0.5 else 1, 1, y.reshape(y.size, 1))


def load_data():
    mnist = input_data.read_data_sets("MNIST_data/")
    train_x, train_y, test_x, test_y = mnist.train.images, mnist.train.labels, \
                                       mnist.test.images, mnist.test.labels
    train_y = tran2logit(train_y)
    test_y = tran2logit(test_y)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    model = LogisticRegressionModel()
    model.fit(train_x, train_y)

    predict_y = np.around(model.predict(test_x))
    print "在测试数据上的表现:accuracy:%f" % cal_pr(test_y, predict_y)
    # show_data()






