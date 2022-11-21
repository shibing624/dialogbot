# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest
from transformers import BertTokenizerFast, GPT2LMHeadModel, AutoTokenizer,AutoModelWithLMHead
import sys

sys.path.append('..')
from dialogbot.kg import question_parser
from dialogbot import GPTBot


class QParseCase(unittest.TestCase):
    def test_something(self):
        handler = question_parser.QuestionPaser()
        print(handler)

        bot = GPTBot()
        msgs = ['苹果手机怎么卖？',
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
            r = bot.answer(msg)
            print('gpt2', msg, r)


if __name__ == '__main__':
    unittest.main()
