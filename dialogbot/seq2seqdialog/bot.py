# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from collections import deque

from dialogbot.seq2seqdialog.model import Model
from dialogbot.utils.logger import logger


class Seq2SeqBot:
    def __init__(self, vocab_path,
                 model_dir_path,
                 last_txt_len=100):
        self.last_txt = deque([], last_txt_len)
        self.m = Model(vocab_path=vocab_path, model_dir_path=model_dir_path)

    def answer(self, query):
        self.last_txt.append(query)
        logger.debug('-' * 20)
        logger.debug("init_query=%s" % query)
        response = self.m.predict(query)
        logger.debug("seq2seq_response=%s" % response)
        self.last_txt.append(response)
        return response
