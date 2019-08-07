# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from dialogbot import Bot

bot = Bot()

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
