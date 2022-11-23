# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import sys

sys.path.append('..')

from dialogbot import GPTBot

if __name__ == '__main__':
    # 具体的指定GPT2生成对话模型
    bot = GPTBot()
    msgs = [
        '你吃了吗？',
        '明天晚上能发出来吗?',
        '有5元 的 东西 吗? 哪种口味好吃',
        '这个金额是否达到包邮条件?',
        '好的谢谢哦。',
        '好的谢了',
        '你好苹果怎么卖？',
        '你的病今天好点了吗？',
        '你今天好点了吗？',
    ]
    for msg in msgs:
        r = bot.answer(msg, use_history=False)
        print('gpt2', msg, r)
