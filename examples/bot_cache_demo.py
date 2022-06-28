# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from loguru import logger
import sys

sys.path.append('..')
from dialogbot import Bot
from dialogbot.utils.io import save_json, load_json



class BotServer:
    def __init__(self, cache_path='cache.json'):
        self.bot = Bot()
        self.cache = {}
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            self.cache = load_json(cache_path)
            logger.info("use cache, cache file: %s" % cache_path)

    def answer(self, query):
        if query in self.cache:
            response = self.cache[query]
        else:
            response = self.bot.answer(query)
            self.cache[query] = response
            if self.cache_path:
                save_json(self.cache, self.cache_path)
                logger.info("save cache.")
        return response

if __name__ == '__main__':
    # Batch queries
    server = BotServer()
    query_list = [
        "王者荣耀哪个英雄最秀",
        "姚明有多高？",
        "姚明老婆是谁",
        "北京天气",
        "上海天气",
        "雅阁现在多少钱",
        "王者荣耀哪个英雄最贵？",
        "百日咳什么症状？",
        "百日咳要治疗多久？",
        "百日咳不能吃啥？",
        "介绍一下百日咳",
        "百日咳用啥药物？",
        "百日咳的预防措施有哪些？",
    ]
    for i in query_list:
        r = server.answer(i)
        print(i, r)
    while True:
        sys.stdout.flush()
        input_text = input("user:")
        if input_text == 'q':
            break
        print("chatbot:", server.answer(input_text))
