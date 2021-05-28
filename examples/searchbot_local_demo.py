# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot.searchdialog.bot import SearchBot

if __name__ == '__main__':

    tfidfbot = SearchBot(question_answer_path='../dialogbot/data/taobao/question_answer.txt',
                         vocab_path='../dialogbot/data/taobao/vocab.txt',
                         search_model="tfidf")
    onehotbot = SearchBot(question_answer_path='../dialogbot/data/taobao/question_answer.txt',
                          vocab_path='../dialogbot/data/taobao/vocab.txt',
                          search_model="onehot")
    bm25bot = SearchBot(question_answer_path='../dialogbot/data/taobao/question_answer.txt',
                        vocab_path='../dialogbot/data/taobao/vocab.txt',
                        search_model="bm25")
    msgs = ['明天晚上能发出来吗?',
            '有5元的东西吗? 哪种口味好吃',
            '这个金额是否达到包邮条件',
            '好的谢谢哦。',
            '好的谢了']
    for msg in msgs:
        search_response, sim_score = tfidfbot.answer(msg, mode='qa')
        print('tfidfbot', msg, search_response, sim_score)

        search_response, sim_score = onehotbot.answer(msg, mode='qa')
        print('onehotbot', msg, search_response, sim_score)

        search_response, sim_score = bm25bot.answer(msg, mode='qa')
        print('bm25bot', msg, search_response, sim_score)
