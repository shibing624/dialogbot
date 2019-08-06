# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot import Bot
bot = Bot()

if __name__ == "__main__":
    print(bot.answer('这车怎么卖？'))
