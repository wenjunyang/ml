#!/usr/bin/python2
# -*-coding:utf-8-*-
import numpy as np


def feature_normalize(x):
    n_c = x.shape[1]
    for i in range(n_c):
        m = np.mean(x[:, i])
        s = np.std(x[:, i])
        x[:, i] = (x[:, i] - m) if s == 0.0 else (x[:, i] - m) / s
    return np.c_[x, np.ones(x.shape[0])]


class LinearRegressionModel:

    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None

    @staticmethod
    def compute_cost(prediction, y):
        sq_error = (y - prediction)
        loss = sq_error.T.dot(sq_error)
        return np.sqrt(loss / y.size)

    def fit(self, x, y, alpha=0.00000008, num_iters=100002):
        self.x = feature_normalize(x)
        self.y = y.reshape(y.size, 1)
        m, d = self.x.shape
        self.theta = np.random.random((d, 1))

        for i in range(1, num_iters):
            gradient = self.x.T.dot(self.x.dot(self.theta) - self.y)
            self.theta -= alpha * gradient
            prediction = self.x.dot(self.theta)
            loss = self.compute_cost(prediction, self.y)
            if i % 100 == 1:
                print "step %d, loss:%f" % (i, loss)

    def predict(self, x):
        return feature_normalize(x).dot(self.theta)


def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    return data[:, :-1], data[:, -1]

if __name__ == '__main__':
    train_x, train_y = read_data("data/housing/housing_train.txt")
    model = LinearRegressionModel()
    model.fit(train_x, train_y)

    test_x, test_y = read_data("data/housing/housing_test.txt")
    predict_y = model.predict(test_x)
    print "测试集的平均误差值:%f" % LinearRegressionModel.compute_cost(predict_y, test_y.reshape(test_y.shape[0], 1))