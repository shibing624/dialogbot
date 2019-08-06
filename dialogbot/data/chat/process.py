# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from codecs import open
from collections import Counter


def build_dict(train_paths, vocab_path):
    word_count = Counter()
    for train_path in train_paths:
        with open(train_path, "r", encoding="utf-8") as f:
            cnt = 0
            for line in f:
                q, a = line.strip().split('\t')
                q_w = q.strip().split(' ')
                a_w = a.strip().split(' ')
                word_count.update(q_w)
                word_count.update(a_w)
                cnt += 1
                if cnt % 10000 == 0:
                    print('file: %s , count size: %d' % (train_path, cnt))
            print('ok, file: %s , count size: %d' % (train_path, cnt))

    with open(vocab_path, "w", encoding="utf-8") as fw:
        word_cnt_list = list(word_count.items())
        word_cnt_list.sort(key=lambda x: x[1], reverse=True)
        for word, count in word_cnt_list:
            fw.write("{}\t{}\n".format(word, count))
        print('build file: %s ok, count size: %d' % (vocab_path, len(word_cnt_list)))


if __name__ == '__main__':
    build_dict(['qa_all.tsv'], 'vocab.txt')
