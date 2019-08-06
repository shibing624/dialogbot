# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot.searchdialog.bot import SearchBot

bm25bot = SearchBot(question_answer_path='../dialogbot/data/taobao/question_answer.txt',
                    vocab_path='../dialogbot/data/taobao/vocab.txt',
                    search_model="bm25")
msgs = ['明天晚上能发出来吗?',
        '有5元的东西吗? 哪种口味好吃',
        '这个金额是否达到包邮条件',
        '好的谢谢哦。',
        '好的谢了']
for msg in msgs:
    search_response, sim_score = bm25bot.answer(msg, mode='qa')
    print('bm25bot', msg, search_response, sim_score)

while True:
    print("input text:")
    msg = input()
    r, s = bm25bot.answer(msg)
    print(r, s)
