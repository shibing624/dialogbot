# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from chatbot import config
from chatbot.searchdialog.bot import SearchBot
from chatbot.util.tokenizer import Tokenizer

msg = '你好，荔枝是怎么卖的？'
msg_tokens = Tokenizer.tokenize(msg, True)
bot = SearchBot(question_answer_path=config.question_answer_path,
                context_response_path=config.context_response_path,
                vocab_path=config.vocab_path,
                search_model="tfidf",)
search_response, sim_score = bot.search(msg_tokens, mode='qa')
print(search_response, sim_score)

bot = SearchBot(question_answer_path=config.question_answer_path,
                context_response_path=config.context_response_path,
                vocab_path=config.vocab_path,
                search_model="vector",)
search_response, sim_score = bot.search(msg_tokens, mode='qa')
print(search_response, sim_score)