#!/usr/bin/python2
# -*-coding:utf-8-*-

"""
一个简单的CRF实现,特征函数使用如下固定的模板函数

# tag  B M E S

# Unigram
U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,0]
U04:%x[2,0]
U05:%x[-2,0]/%x[-1,0]/%x[0,0]
U06:%x[-1,0]/%x[0,0]/%x[1,0]
U07:%x[0,0]/%x[1,0]/%x[2,0]
U08:%x[-1,0]/%x[0,0]
U09:%x[0,0]/%x[1,0]

# Bigram
B
"""
from __future__ import division
import codecs
from datetime import datetime

import numpy as np


def current_time_str():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def print_with_time(info):
    print "%s %s" % (current_time_str(), info)


class StatusFeatureFunction:

    def __init__(self, row_offsets, o, s):
        self.row_offsets = row_offsets
        self.o = o
        self.s = s

    def cal_total_feature(self, observe, status):
        if len(observe) != len(status):
            raise Exception("观察序列:%s与状态序列:%s不匹配")
        result = 0
        for i in range(len(status)):
            result += self.cal_current_feature(i, observe, status[i])

        return result

    def cal_current_feature(self, i, observe, cur_s, pre_s=None):
        observe_condition = False
        for offset in self.row_offsets:
                index = i + offset
                if index >= len(observe):
                    current_o = CRFChineseSeg.OBSERVE_END
                elif index <= 0:
                    current_o = CRFChineseSeg.OBSERVE_START
                else:
                    current_o = observe[index]

                if current_o == self.o:
                    observe_condition = True
                    break
        return 1 if observe_condition and cur_s == self.s else 0


class TransFeatureFunction:
    def __init__(self, o, pre_s, s):
        self.o = o
        self.pre_s = pre_s
        self.s = s

    def cal_total_feature(self, observe, status):
        if len(observe) != len(status):
            raise Exception("观察序列:%s与状态序列:%s不匹配")
        result = 0
        for i in range(len(status) + 1):
            pre_s = status[i-1] if i > 0 else CRFChineseSeg.STATUS_START
            cur_s = status[i] if i < len(status) else CRFChineseSeg.STATUS_END
            result += self.cal_current_feature(i, observe, cur_s, pre_s)
        return result

    def cal_current_feature(self, i, observe, cur_s, pre_s):
        cur_o = observe[i] if i < len(observe) else CRFChineseSeg.OBSERVE_END
        return 1 if self.pre_s == pre_s and self.s == cur_s and cur_o == self.o else 0



class CRFChineseSeg:

    OBSERVE_START = -1
    OBSERVE_END = -2
    STATUS_START = -1
    STATUS_END = -2

    def __init__(self, train_file):
        self.observe_list = None          # 观察值集合,即字的个数,假设个数为:N
        self.status_list = None           # 状态集合,即标注的种类,假设个数为:L
        self.feature_func_list = None     # 特征函数个数,假设个数为:T=t1*N*L+t2*N*L*L
                                          # ,其中t1,t2分别为unigram,bigram模板条目数
        self.train_feature = None         # 训练集的关于特征函数的特征值:n*T,n为训练集行数
        self.relaxation = 0               # 松弛值,取最大值的1000倍
        self.train_file = train_file      # 训练文件
        self.train_data = None            # 训练数据,个数:n
        self.tag = None                   # 训练标签,个数:n
        self.actual_expect = None         # 各个特征的实际期望值:T
        self.train_matrix = None          # 每条训练数据的没有乘系数的矩阵形式, n*(len(x)+1)*L*L*T
        self.weight = None                # 每个特征的权重:T

    def _init_feature_function(self):
        offsets_list = [
            [-2],
            [-1],
            [0],
            [1],
            [2],
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2],
            [-1, 0],
            [0, 1]
        ]
        self.feature_func_list = []
        for offsets in offsets_list:
            for o in self.observe_list:
                for s in self.status_list:
                    self.feature_func_list.append(StatusFeatureFunction(offsets, o, s))

        for o in self.observe_list:
            for pre_s in self.status_list:
                for s in self.status_list:
                    self.feature_func_list.append(TransFeatureFunction(o, pre_s, s))

    def train(self):

        with codecs.open(self.train_file, "r", "utf-8") as target:
            lines = target.readlines()

        lines = map(lambda l: l.strip(), lines)

        print_with_time("读取文件完成,总共%d行" % len(lines))

        self._init_list(lines)
        print_with_time("观察值集合大小:%d,状态集合大小:%d" %
                        (len(self.observe_list), len(self.status_list)))

        print_with_time("开始构造所有特征函数..")
        self._init_feature_function()
        print_with_time("特征函数个数:%d" % len(self.feature_func_list))


        # 切分训练数据,计算特征向量
        print_with_time("开始切分训练数据,计算特征向量")
        self._init_train_feature(lines)

        # 计算训练集中每一个特征的实际期望值
        print_with_time("开始计算每个特征的实际期望值")
        self._cal_actual_expect()

        # 计算训练集中每个特征关于条件概率函数
        print_with_time("开始计算每个特征关于条件转移概率函数的特征值")
        self._cal_train_matrix()

        # 计算权重
        print_with_time("开始迭代计算权重")
        self._cal_weight()

    def _cal_weight(self):
        self.weight = np.zeros(len(self.feature_func_list))
        for step in range(100):
            print_with_time("第%00d次迭代" % step)
            expects = []
            for i in range(len(self.train_data)):
                expects.append(self._cal_single_expect(self.train_data[i], self.train_matrix[i]))
            sum_expects = np.sum(expects, axis=0)
            self.weight = np.log(sum_expects / self.actual_expect) / self.relaxation

    def _cal_single_expect(self, observe, feature_matrix):
        # 计算M矩阵
        weighting_matrix = []
        for i in range(len(observe) + 1):
            cur_matrix = []
            for j in range(len(self.status_list)):
                cur_vec = []
                for k in range(len(self.status_list)):
                    cur_vec.append(np.exp(np.dot(self.weight, feature_matrix[i][j][k])))
                cur_matrix.append(cur_vec)
            weighting_matrix.append(cur_matrix)

        # 计算alpha矩阵
        alpha_matrix = []

        init_alpha = map(lambda x: 1 if x == self.STATUS_START else 0, self.status_list)

        for i in range(len(observe)):
            if i == 0:
                last_alpha = init_alpha
            else:
                last_alpha = alpha_matrix[i-1]
            alpha_matrix.append(np.dot(last_alpha, weighting_matrix[i]))

        # 计算beta矩阵
        beta_matrix = range(len(observe) + 1)
        beta_matrix[len(observe)] = map(lambda x: 1 if x == self.STATUS_END else 0,
                                        self.status_list)
        for i in range(len(observe) - 1, -1, -1):
            beta_matrix[i] = np.dot(weighting_matrix[i + 1], beta_matrix[i + 1])

        # 计算Z(x)
        normalize_z = sum(alpha_matrix[len(observe) - 1])

        # 计算期望值
        expect = []
        for fun in self.feature_func_list:
            cur_expect = 0
            for i in range(len(observe) + 1):
                sum_tk = 0
                for j in range(len(self.status_list)):
                    for k in range(len(self.status_list)):
                        tk = fun.cal_current_feature(i, observe, self.status_list[j], self.status_list[k])
                        sum_tk += tk
                if i == 0:
                    alpha_vec = init_alpha
                else:
                    alpha_vec = alpha_matrix[i-1]
            non_normalize = np.dot(np.dot(alpha_vec, weighting_matrix[i]), beta_matrix[i])

            expect.append(sum_tk * non_normalize / normalize_z)
        return expect

    def _cal_train_matrix(self):
        self.train_matrix = []
        for observe in self.train_data:
            self.train_matrix.append(self._cal_trans_matrix(observe))

    def _cal_trans_matrix(self, observe):
        matrix_list = []
        for i in range(len(observe) + 1):
            cur_matrix = []
            for pre_s in self.status_list:
                cur_vec = []
                for cur_s in self.status_list:
                    feature = []
                    for fun in self.feature_func_list:
                        feature.append(fun.cal_current_feature(i, observe, cur_s, pre_s))
                    cur_vec.append(feature)
                cur_matrix.append(cur_vec)
            matrix_list.append(cur_matrix)
        return matrix_list

    def _cal_actual_expect(self):
        expect = []
        for i in range(len(self.feature_func_list)):
            expect.append(sum(map(lambda x: x[i], self.train_feature)))
        self.actual_expect = np.array(expect)

    def _cal_feature(self, observe, status):
        if len(observe) != len(status):
            raise Exception("观察序列:%s与状态序列:%s不匹配")
        feature = []
        for feature_fun in self.feature_func_list:
            feature.append(feature_fun.cal_total_feature(observe, status))
        return feature

    def _init_train_feature(self, lines):
        self.train_data = []
        self.train_feature = []
        self.tag = []
        current_observe = []
        current_status = []
        for line in lines + [""]:
            if line:
                items = line.split()
                current_observe.append(items[0])
                current_status.append(items[1])
            else:
                if len(current_status) > 0:
                    print_with_time("计算第%d个训练数据的特征向量" % len(self.train_data))
                    self.train_data.append(current_observe)
                    self.tag.append(current_status)
                    self.train_feature.append(self._cal_feature(current_observe, current_status))
                current_observe = []
                current_status = []
        # 暂定一个松弛值
        self.relaxation = 1000 * max(map(sum, self.train_feature))

    def _init_list(self, lines):
        # 确定状态及tag集合
        observe_set = set()
        status_set = set()
        for line in lines:
            if not line:
                continue
            else:
                items = line.split()
                if len(items) != 2:
                    raise Exception("Error occurred in line: %s", line)
                observe_set.add(items[0])
                status_set.add(items[1])
        self.observe_list = [self.OBSERVE_START] + list(observe_set) + [self.OBSERVE_END]
        self.status_list = [self.STATUS_START] + list(status_set) + [self.STATUS_END]


if __name__ == '__main__':
    model = CRFChineseSeg("data/icwb2-data/training/pku_training_crf.utf8")
    model.train()
    print model.weight









