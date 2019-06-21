# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from dialogbot import config
from dialogbot.bot import Bot

bot = Bot(vocab_path=config.vocab_path,
          dialog_mode=config.dialog_mode,
          search_model=config.search_model,
          question_answer_path=config.question_answer_path,
          context_response_path=config.context_response_path)

msgs = ['明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃', '这个 金额 是否 达到 包邮 条件', '好的谢谢哦。', '好的谢了']
for msg in msgs:
    response = bot.answer(msg, use_task=True)
    print("query:", msg)
    print("response:", response)
