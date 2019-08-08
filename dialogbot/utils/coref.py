# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from collections import OrderedDict

from dialogbot.utils.tokenizer import posseg


def pos_demo():
    a = ['我的姐姐给我她的狗。很喜欢.',
         '张先生走过来，给大家看他的新作品', '张林走过来，给大家看他的新作品',
         '今天天气如何？那明天呢？', '北京今天的天气如何？', '上海呢？', '武汉呢？', '武汉明天呢？',
         '范冰冰是谁？', '她男友呢',
         '现任美国总统是谁', '那前任呢',
         '中国的人口是多少？', '那美国的GDP呢',

         '王伟喜欢那里', '那我呢',

         '小明喜欢哪里？', '那我呢'
         ]
    for i in a:
        b = posseg.lcut(i)
        print(i, b)


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
get_coref_sentence(a[0], a[1])
