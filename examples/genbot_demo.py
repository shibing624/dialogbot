# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:  test seq2seq model
"""

import sys

sys.path.append('..')

from dialogbot import GPTBot
from dialogbot import Bot

if __name__ == '__main__':
    bot = Bot()
    response = bot.answer('亲 你吃了吗？', use_gen=True, use_search=False, use_task=False)
    print(response)

    # 具体的指定GPT2生成对话模型
    bot = GPTBot()
    msgs = ['你吃了吗？', '明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃',
            '这个金额是否达到包邮条件?', '好的谢谢哦。', '好的谢了', '你好苹果怎么卖？']
    for msg in msgs:
        r = bot.answer(msg)
        print('gpt2', msg, r)
