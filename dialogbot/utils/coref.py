# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from collections import OrderedDict

from dialogbot.utils.tokenizer import posseg


def keep_word_tags(query, tags=['n']):
    result = OrderedDict()
    idx = 0
    for i in posseg.cut(query):
        # 只保留所需词性的词
        if i.flag in tags:
            result[idx] = i
        idx += 1
    return result


semantic_tags = ['n', 'f', 'nr', 'i', 'ns', 'nt', 'nz', 'q', 'r', 's', 'tg', 't', 'vg', 'v', 'vd', 'vn', 'w']


class Semantic:
    def __init__(self, sentence):
        self.sentence = sentence
        self.entity = keep_word_tags(sentence)
        self.semantic_words = keep_word_tags(sentence, semantic_tags)


if __name__ == '__main__':
    def get_coref_sentence(gold_sentence, sentence):
        gold = Semantic(gold_sentence)
        query = Semantic(sentence)
        print(gold.entity, gold.semantic_words)
        print(query.entity, query.semantic_words)
        gold.semantic_words.pop(0)
        s = ''
        for k, v in query.semantic_words.items():
            s += v.word

        g = ''
        for k, v in gold.semantic_words.items():
            g += v.word
        new_s = s + g
        print(new_s)


    a = ['现任美国总统是谁?', '那,前任呢', '泰国呢', ]
    for i in a:
        b = posseg.lcut(i)
        print(i, b)
    get_coref_sentence(a[0], a[1])
