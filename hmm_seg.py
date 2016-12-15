# -*- coding: utf-8 -*-
#! /usr/bin/env python

from __future__ import division

import codecs
import math
import numpy as np
import sys


class HMMChineseSeg:

    _STATUS = (_B, _M, _E, _S) = range(4)
    _DEFAULT_LOG_PROB = -3.14e+100

    def __init__(self):
        self.emit_map = map(lambda x: {}, self._STATUS)
        self.trans_matrix = map(lambda x: map(lambda y: 0, self._STATUS), self._STATUS)
        self.init_status = map(lambda x: 0, self._STATUS)

    @staticmethod
    def _inc_value(m, key):
        cnt = m.setdefault(key, 0)
        m[key] = cnt + 1

    def add_train_line(self, line):
        items = filter(lambda x: len(x) > 0, line.split())
        if len(items) == 0:
            return

        status = []

        # 转换成状态
        for item in items:
            if len(item) == 1:
                status.append(self._S)
            else:
                status.append(self._B)
                for i in range(1, len(item) - 1):
                    status.append(self._M)
                status.append(self._E)

        pairs = zip("".join(items), status)
        self.init_status[pairs[0][1]] += 1
        self._inc_value(self.emit_map[pairs[0][1]], pairs[0][0])

        for i in range(len(pairs)):
            this_pair = pairs[i]
            last_pair = pairs[i-1]
            self.trans_matrix[last_pair[1]][this_pair[1]] += 1
            self._inc_value(self.emit_map[this_pair[1]], this_pair[0])

    def _cal_prob(self):

        print "trans matrix"
        for trans in self.trans_matrix:
            print ','.join(map(lambda x: str(x), trans))
        print "init status"
        print ",".join(map(lambda x: str(x), self.init_status))

        # 计算emit概率的log值
        for emit in self.emit_map:
            total_cnt = sum(emit.values())
            for key in emit.keys():
                emit[key] = self._DEFAULT_LOG_PROB if emit[key] == 0 \
                    else math.log(emit[key] / total_cnt)

        # 计算转移概率的log值
        for trans in self.trans_matrix:
            total = sum(trans)
            for i in range(len(trans)):
                trans[i] = self._DEFAULT_LOG_PROB if trans[i] == 0 \
                    else math.log(trans[i] / total)

        # 计算初始状态概率的log值
        total = sum(self.init_status)
        for i in range(len(self.init_status)):
            self.init_status[i] = self._DEFAULT_LOG_PROB if self.init_status[i] == 0 \
                else math.log(self.init_status[i] / total)

    def train(self, lines=[]):
        for line in lines:
            self.add_train_line(line)

        self._cal_prob()

    def segment(self, line, seg_char=' '):
        if not line:
            return ""
        path = [map(lambda x: -1, self._STATUS)]
        last_max = []
        for i in range(len(self._STATUS)):
            last_max.append(self.init_status[i] +
                            self.emit_map[i].get(line[0], self._DEFAULT_LOG_PROB))

        for i in range(1, len(line)):
            this_path = map(lambda x: -1, self._STATUS)
            this_max = map(lambda x: self._DEFAULT_LOG_PROB, self._STATUS)

            for m in range(len(self._STATUS)):
                emit_prob = self.emit_map[m].get(line[i], self._DEFAULT_LOG_PROB)
                probs = [last_max[n] + self.trans_matrix[n][m] + emit_prob for n in range(len(self._STATUS))]
                this_path[m] = np.argmax(probs)
                this_max[m] = np.max(probs)

            path.append(this_path)
            last_max = this_max

        last_index = np.argmax(last_max)
        status = []
        for i in range(len(line) - 1, -1, -1):
            status.append(last_index)
            last_index = path[i][last_index]

        status.reverse()
        return self.insert_seg_char(line, status, seg_char)

    def insert_seg_char(self, text, status, seg_char=' '):
        result = ""
        for (c, s) in zip(text, status):
            result += c
            if s in (self._E, self._S):
                result += seg_char
        return result


def interactive_test():
    while True:
        text = raw_input("")
        text = text.decode('utf8').strip()
        result = model.segment(text, '\\')
        print result


def segment_file(in_file, out_file):
    with codecs.open(in_file, 'r', 'utf-8') as target_in, codecs.open(out_file, 'w', 'utf-8') as target_out:
        for line in target_in.readlines():
            result = model.segment(line.strip())
            target_out.write(result + '\n')

if __name__ == '__main__':
    train_path = "data/icwb2-data/training/pku_training.utf8"
    model = HMMChineseSeg()
    with codecs.open(train_path, 'r', 'utf-8') as train_file:
        lines = train_file.readlines()
        for line in lines:
            model.add_train_line(line.strip())

    model.train()

    if len(sys.argv) == 3:
        segment_file(sys.argv[1], sys.argv[2])
    else:
        interactive_test()






