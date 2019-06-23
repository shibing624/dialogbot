# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

from dialogbot import Bot

bot = Bot()
l = ['快递多少天到？', '苹果怎么卖？']
for i in l:
    c = bot.answer(i)
    print("q:%s => a:%s" % (i, c))
