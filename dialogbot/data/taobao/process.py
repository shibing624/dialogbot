# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from codecs import open
from collections import Counter


def build_dict(train_path, single_path, multi_path, vocab_path):
    word_count = Counter()
    with open(train_path, "r", encoding="utf-8") as fr, \
            open(single_path, "w", encoding="utf-8") as fw1, \
            open(multi_path, "w", encoding="utf-8") as fw2:
        cnt = 0
        for line in fr:
            words = line.split()[1:]
            word_count.update(words)
            items = line.strip().split("\t")
            if items[0] == "1":
                dialog = items[1:]
                context = dialog[:-1][::2]
                question = dialog[-2]
                response = dialog[-1]
                fw1.write(question + "\t" + response + "\n")
                fw2.write("<s>".join(context) + "\t" + response + "\n")
                cnt += 1
        print('file: %s , count size: %d' % (train_path, cnt))

    with open(vocab_path, "w", encoding="utf-8") as fw:
        word_cnt_list = list(word_count.items())
        word_cnt_list.sort(key=lambda x: x[1], reverse=True)
        for word, count in word_cnt_list:
            fw.write("{}\t{}\n".format(word, count))
        print('build file: %s ok, count size: %d' % (vocab_path, len(word_cnt_list)))


if __name__ == '__main__':
    build_dict('dev.txt', 'question_answer.txt', 'context_response.txt', 'vocab.txt')
