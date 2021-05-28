# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot import Bot

if __name__ == '__main__':
    bot = Bot()
    msgs = [
        '亲 吃了吗',
        '雅阁车二手卖多少钱？',
        '北京天气?',
        '上海的天气呢',
        '明天晚上能发出来吗?',
        '有5元的东西吗? 哪种口味好吃',
        '这个金额是否达到包邮条件',
        '好的谢谢哦。',
        '姚明多高?',
        '长城哪一年开始修的?',
    ]
    for msg in msgs:
        print("query:", msg)
        response = bot.answer(msg)
        print("response:", response)
