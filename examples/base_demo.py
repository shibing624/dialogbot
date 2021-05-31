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
    response = bot.answer('亲 吃了吗？')
    print(response)
