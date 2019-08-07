# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot import Bot

bot = Bot()

msgs = ['明天晚上能发出来吗?',
        '有5元的东西吗? 哪种口味好吃',
        '这个金额是否达到包邮条件',
        '好的谢谢哦。',
        '好的谢了',
        '姚明多高?',
        '长城哪一年开始修的?', ]
for msg in msgs:
    response = bot.answer(msg, use_task=True)
    print("query:", msg)
    print("response:", response)

