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
        '好的谢了']
for msg in msgs:
    response = bot.answer(msg, use_task=True)
    print("query:", msg)
    print("response:", response)


def start_dialog():
    print("\nChatbot: %s\n" % "您好，我是可爱的对话机器人小智，有问题都可以向我提问哦~")
    print("input1: ", end="")

    while True:
        msg = input().strip()
        if msg.lower() == "finish":
            print("Chatbot: %s\n\n" % "change session", end="")
            print("input1: ", end="")
            bot.set_context([])
        elif msg.lower() == "exit":
            print("Chatbot: %s\n\n" % "感谢您的支持，我们下次再见呢~, 拜拜亲爱哒")
            exit()
        else:
            bot.context.append(msg)
            response = bot.answer(msg, use_task=True)
            print("output%d: %s\n\n" % (len(bot.context) / 2, response), end="")
            print("input%d: " % (len(bot.context) / 2 + 1), end="")
            bot.context.append(response)


start_dialog()
