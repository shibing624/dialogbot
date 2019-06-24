#!/usr/bin/env python3
# coding: utf-8
# File: chatbot_graph.py
# Author: XuMing（xuming624@qq.com),lhy<lhy_in_blcu@126.com>
import sys

sys.path.append('..')
from dialogbot.kg.bot import KGBot

if __name__ == '__main__':
    handler = KGBot()
    print("eg：百日咳什么症状？ 或 百日咳要治疗多久？ 或 百日咳不能吃啥？ 或 百日咳可以吃啥？ 或 "
          "介绍一下百日咳 或 百日咳用啥药物？或 百日咳怎么检查？或 百日咳的预防措施有哪些？")
    while True:
        question = input('query:')
        if question == 'quit':
            print('quit.')
            break
        answer = handler.answer(question)
        print('answer: %s' % answer)
