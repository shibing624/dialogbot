# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

# test evaluate
from dialogbot import config

from dialogbot.seq2seqdialog.model import Model

seq2seq_inst = Model(config.vocab_path, config.seq2seq_model_path)
user_msgs = '你好 苹果 怎么 卖 ？'
response = seq2seq_inst.predict(user_msgs)
print('response:', response)
b_score = seq2seq_inst.evaluate(config.dialog_mode, config.predict_result_path, config.question_answer_path)
print(b_score)
# c_score = seq2seq_inst.evaluate(config.dialog_mode, config.predict_result_path, config.context_response_path)
# print(c_score)



# test Seq2SeqBot
from dialogbot.seq2seqdialog.bot import Seq2SeqBot

bot = Seq2SeqBot(vocab_path=config.vocab_path, model_dir_path=config.seq2seq_model_path)
msgs = ['明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃', '这个 金额 是否 达到 包邮 条件', '好的谢谢哦。', '好的谢了', '你好 苹果 怎么 卖 ？']
for msg in msgs:
    r = bot.answer(msg)
    print('seq2seq', msg, r)
