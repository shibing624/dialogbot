# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from dialogbot import config
from dialogbot.seq2seqdialog.bot import Seq2SeqBot

bot = Seq2SeqBot(vocab_path=config.vocab_path)
msgs = ['明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃', '这个 金额 是否 达到 包邮 条件', '好的谢谢哦。', '好的谢了']
for msg in msgs:
    r = bot.answer(msg)
    print('seq2seq', msg, r)
