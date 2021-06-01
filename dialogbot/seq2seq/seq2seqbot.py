# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from collections import deque

from .model import Model
from ..utils.log import logger


class Seq2SeqBot:
    def __init__(self, vocab_path,
                 model_dir_path,
                 last_txt_len=100):
        self.last_txt = deque([], last_txt_len)
        self.vocab_path = vocab_path
        self.model_dir = model_dir_path
        self.model = None

    def init(self):
        if not self.model:
            if os.path.exists(self.vocab_path) and os.path.exists(self.model_dir):
                self.model = Model(vocab_path=self.vocab_path, model_dir_path=self.model_dir)
            else:
                logger.warning("Seq2Seq model not found. vocab: {}, model: {}".format(self.vocab_path, self.model_dir))

    def answer(self, query):
        self.init()
        response = ''
        if not self.model:
            return response
        self.last_txt.append(query)
        logger.debug('-' * 20)
        logger.debug("init_query=%s" % query)
        response = self.model.predict(query)
        logger.debug("seq2seq_response=%s" % response)
        self.last_txt.append(response)
        return response
