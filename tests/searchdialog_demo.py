# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from dialogbot import config
from dialogbot.searchdialog.bot import SearchBot

tfidfbot = SearchBot(search_model="tfidf")
onehotbot = SearchBot(question_answer_path=config.question_answer_path,
                      context_response_path=config.context_response_path,
                      vocab_path=config.vocab_path,
                      search_model="onehot")

bm25bot = SearchBot(question_answer_path=config.question_answer_path,
                    context_response_path=config.context_response_path,
                    vocab_path=config.vocab_path,
                    search_model="bm25")
vectorbot = SearchBot(question_answer_path=config.question_answer_path,
                      context_response_path=config.context_response_path,
                      vocab_path=config.vocab_path,
                      search_model="vector")
msgs = ['明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃', '这个 金额 是否 达到 包邮 条件', '好的谢谢哦。', '好的谢了']
for msg in msgs:
    search_response, sim_score = tfidfbot.answer(msg, mode='qa')
    print('tfidfbot', msg, search_response, sim_score)

    search_response, sim_score = onehotbot.answer(msg, mode='qa')
    print('onehotbot', msg, search_response, sim_score)

    search_response, sim_score = bm25bot.answer(msg, mode='qa')
    print('bm25bot', msg, search_response, sim_score)

    search_response, sim_score = vectorbot.answer(msg, mode='qa')
    print('vectorbot', msg, search_response, sim_score)
