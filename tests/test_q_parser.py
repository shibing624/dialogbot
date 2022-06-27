# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import unittest
import sys

sys.path.append('..')
from dialogbot.kg import question_parser


class QParseCase(unittest.TestCase):
    def test_something(self):
        handler = question_parser.QuestionPaser()
        print(handler)


if __name__ == '__main__':
    unittest.main()
