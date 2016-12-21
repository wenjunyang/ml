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
import collections
from datetime import datetime

import math

import numpy as np


def current_time_str():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def print_with_time(info):
    print "%s %s" % (current_time_str(), info)


class FeatureFunction:

    OBSERVE_START = -1
    OBSERVE_END = -2
    STATUS_START = -1
    STATUS_END = -2

    def __init__(self, observes, labels):
        """
        根据字典及标签构造特征函数
        :param observes: 字典集合
        :param labels: 标签集合
        :return:
        """
        self.funs = []

        self.status_idx_maps = []     # 状态特征函数索引的映射
        self.tran_idx_maps = {}        # 转移特征函数索引映射

        self.unigram = [(-2,), (-1,), (0,), (1,), (2,), (-2, -1, 0)]

        # 字典及标签分别加入开始
        self.words = observes
        self.labels = labels

        self.status_fun_num = 0
        self.tran_fun_num = 0

        # unigram
        for u in self.unigram:
            status_map = {}
            for w in self.words + [self.OBSERVE_START, self.OBSERVE_END]:
                for l in self.labels + [self.STATUS_END]:
                    self.funs.append((u, w, l))
                    status_map[(w, l)] = self.status_fun_num
                    self.status_fun_num += 1
            self.status_idx_maps.append(status_map)

        # bigram 使用默认的一个
        for w in self.words + [self.OBSERVE_END]:
            for pre_l in self.labels + [self.STATUS_START]:
                for cur_l in self.labels + [self.STATUS_END]:
                    self.funs.append((w, pre_l, cur_l))
                    self.tran_idx_maps[(w, pre_l, cur_l)] = self.status_fun_num + self.tran_fun_num
                    self.tran_fun_num += 1

    def _unigram_condition(self, u, x, i):
        words = []
        for offset in u:
            idx = i + offset
            if idx < 0:
                word = self.OBSERVE_START
            elif idx >= len(x):
                word = self.OBSERVE_END
            else:
                word = x[idx]
            words.append(word)
        return words

    def cal_single_feature_single_position(self, fun_idx, pre_y, cur_y, x, i):
        if fun_idx < 0 or fun_idx >= len(self.funs):
            raise Exception("特征函数索引越界")
        elif fun_idx < self.status_fun_num:
            (u, w, l) = self.funs[fun_idx]
            return 1 if cur_y == l and w in self._unigram_condition(u, x, i) else 0
        else:
            (w, pre_l, cur_l) = self.funs[fun_idx]
            return 1 if pre_y == pre_l and cur_y == cur_l \
                        and w in self._unigram_condition((0,), x, i) else 0

    def cal_single_feature(self, fun_idx, x, y):
        result = 0
        if len(x) != len(y):
            raise Exception("观测数据和状态数据不是相同长度")

        for i in range(len(x) + 1):
            pre_y = self.STATUS_START if i == 0 else y[i-1]
            cur_y = self.STATUS_END if i == len(y) else y[i]
            result += self.cal_single_feature(fun_idx, pre_y, cur_y, x, i)
        return result

    def cal_single_position(self, pos, x, pre_y, cur_y):
        if pos < 0 or pos > len(x):
            raise Exception("索引不是合适范围")
        else:
            non_zero_idx = []
            for i in range(len(self.status_idx_maps)):
                status_map = self.status_idx_maps[i]
                u = self.unigram[i]
                words = self._unigram_condition(u, x, pos)
                for word in words:
                    fun_idx = status_map.get((word, cur_y))
                    if fun_idx:
                        non_zero_idx.append(fun_idx)
                fun_idx = self.tran_idx_maps.get((word, pre_y, cur_y))
                if fun_idx:
                    non_zero_idx.append(fun_idx)
            return non_zero_idx

    def cal_feature(self, x, y):
        if len(x) != len(y):
            raise Exception("观测数据和状态数据不是相同长度")
        idxs = []
        for i in range(len(x) + 1):
            pre_y = self.STATUS_START if i == 0 else y[i-1]
            cur_y = self.STATUS_END if i == len(y) else y[i]
            idxs += self.cal_single_position(i, x, pre_y, cur_y)
        return collections.Counter(idxs)

    def feature_num(self):
        return self.status_fun_num + self.tran_fun_num


class CRFChineseSeg:

    def __init__(self, train_file):
        self.train_file = train_file
        self.train_x = []
        self.train_y = []
        self.observe_list = None
        self.status_list = None
        self.observe_map = None
        self.status_list = None
        self.feature_fun = None
        self.actual_expect = None

        self.w = None

    def train(self):
        with codecs.open(self.train_file, "r", "utf-8") as target:
            lines = target.readlines()

        lines = map(lambda l: l.strip(), lines)
        print_with_time("读取文件完成,总共%d行" % len(lines))

        self._init_set_feature_fun(lines)
        self._init_train(lines)
        print_with_time("解析文件完成,观察集合大小:{},状态集合大小:{},训练样本数:{},特征函数个数:{}"
                        .format(len(self.observe_list), len(self.status_list),
                                len(self.train_x), self.feature_fun.feature_num()))

        self._cal_actual_expect()
        print_with_time("实际特征值期望值计算完毕,其中不为0的个数:{},为0的个数:{}"
                        .format(sum(map(lambda x: 1 if x != 0 else 0, self.actual_expect)),
                                sum(map(lambda x: 1 if x == 0 else 0, self.actual_expect))))

    def _init_set_feature_fun(self, lines):
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
        self.observe_list = range(len(observe_set))
        self.status_list = range(len(status_set))
        self.observe_map = dict(zip(observe_set, range(len(observe_set))))
        self.status_map = dict(zip(status_set, range(len(status_set))))
        self.feature_fun = FeatureFunction(self.observe_list, self.status_list)

    def _init_train(self, lines):
        current_x = []
        current_y = []

        for i in range(len(lines)):
            line = lines[i]
            if line:
                items = line.split()
                if len(items) != 2:
                    raise Exception("Error occurred in line: %s", line)
                current_x.append(items[0])
                current_y.append(items[1])

            if (not line) or i == len(lines) - 1:
                if current_x and current_y:
                    self.train_x.append(map(lambda x: self.observe_map[x], current_x))
                    self.train_y.append(map(lambda x: self.status_map[x], current_y))
                    current_x = []
                    current_y = []

    def _cal_actual_expect(self):
        self.actual_expect = [0] * self.feature_fun.feature_num()
        for i in range(len(self.train_x)):
            feature_map = self.feature_fun.cal_feature(self.train_x[i], self.train_y[i])
            for (k, v) in feature_map.iteritems():
                self.actual_expect[k] += v

    def _cal_weight(self):
        self.w = [0.] * self.feature_fun.feature_num()
        print_with_time("开始迭代求权重...")
        step = 0
        while self._iter_one_step():
            step += 1
            print_with_time("第{}步计算完成".format(step))

    def _iter_one_step(self):
        # 计算M矩阵
        for i in range(len(self.train_x)):
            feature = {}
            m_matrixs = []

            # 计算M矩阵及部位0的特征值
            for j in range(len(self.train_x[i])):
                cur_matrix = []
                for pre_y in self.status_list + [FeatureFunction.STATUS_START]:
                    cur_vec = []
                    for cur_y in self.status_list + [FeatureFunction.STATUS_END]:
                        non_zeros = self.feature_fun.cal_single_position(j, self.train_x[j], pre_y, cur_y)
                        feature[(j, pre_y, cur_y)] = non_zeros
                        value = 0
                        for n in non_zeros:
                            value += self.w[n]
                        cur_vec.append(math.exp(value))
                    cur_matrix.append(cur_vec)
                m_matrixs.append(cur_matrix)

            alpha_vecs = [map(lambda x: 0, self.status_list) + [1]]
            for j in range(len(self.train_x[i])):
                alpha_vecs.append()


if __name__ == '__main__':
    model = CRFChineseSeg("data/icwb2-data/training/pku_training_crf.utf8")
    model.train()





