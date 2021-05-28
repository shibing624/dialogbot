#!/usr/bin/env python3
# coding: utf-8
# File: chatbot_graph.py
# Author: XuMing(xuming624@qq.com),lhy<lhy_in_blcu@126.com>
import sys

sys.path.append('..')
from dialogbot.kg.bot import KGBot

if __name__ == '__main__':
    handler = KGBot()
    query_list = ["百日咳什么症状？",
                  "百日咳要治疗多久？",
                  "百日咳不能吃啥？",
                  "介绍一下百日咳",
                  "百日咳用啥药物？",
                  "百日咳的预防措施有哪些？",
                  ]
    for query in query_list:
        answer = handler.answer(query)
        print('answer: %s' % answer)
