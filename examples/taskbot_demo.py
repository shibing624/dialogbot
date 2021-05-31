# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:  test seq2seq model
"""

import sys

sys.path.append('..')

from dialogbot.taskdialog import infer

if __name__ == '__main__':
    q = '这个金额是否达到包邮条件?'
    response = infer.babi.path
    print('response:', response)

    # test task bot

    msgs = ['明天晚上能发出来吗?', '有5元 的 东西 吗? 哪种口味好吃', '这个 金额 是否 达到 包邮 条件', '好的谢谢哦。', '好的谢了', '你好 苹果 怎么 卖 ？']
