#!/usr/bin/env bash

# 隐马尔科夫模型分词效果测试脚本

#分词,存储到中间文件
python hmm_seg.py data/icwb2-data/testing/pku_test.utf8 pku_test_seg.txt

#对比效果,结果直接输出到stdout
data/icwb2-data/scripts/score data/icwb2-data/gold/pku_training_words.utf8 data/icwb2-data/gold/pku_test_gold.utf8 pku_test_seg.txt

#删除临时文件
rm pku_test_seg.txt


