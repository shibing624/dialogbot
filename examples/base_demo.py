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
    response = bot.answer('姚明有多高？')
    print(response)

    # Batch queries
    query_list = [
        "姚明老婆是谁",
        "北京天气",
        "上海天气是谁",
        "雅阁现在多少钱",
        "王者荣耀哪个英雄最秀",
        "百日咳什么症状？",
        "百日咳要治疗多久？",
        "百日咳不能吃啥？",
        "介绍一下百日咳",
        "百日咳用啥药物？",
        "百日咳的预防措施有哪些？",
    ]
    for i in query_list:
        r = bot.answer(i)
        print(i, r)
