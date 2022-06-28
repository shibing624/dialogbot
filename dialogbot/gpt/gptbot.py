# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
from collections import deque
from loguru import logger
from dialogbot import config
from dialogbot.gpt.interact import Inference


class GPTBot:
    def __init__(self, model_dir=config.gpt_model_dir, device="cpu",
                 max_history_len=3, max_len=25, repetition_penalty=1.0, temperature=1.0,
                 topk=8, topp=0.0, last_txt_len=100):
        self.last_txt = deque([], last_txt_len)
        self.model = None
        self.model_dir = model_dir
        self.device = device
        self.max_history_len = max_history_len
        self.max_len = max_len
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.topk = topk
        self.topp = topp

    def init(self):
        if not self.model:
            if os.path.exists(self.model_dir):
                self.model = Inference(self.model_dir, self.device, max_history_len=self.max_history_len,
                                       max_len=self.max_len, repetition_penalty=self.repetition_penalty,
                                       temperature=self.temperature,
                                       topk=self.topk, topp=self.topp)
            else:
                logger.warning("GPT model not found. model: {}".format(self.model_dir))

    def answer(self, query):
        self.init()
        response = ''
        if not self.model:
            return response
        self.last_txt.append(query)
        logger.debug('-' * 20)
        logger.debug("init_query=%s" % query)
        response = self.model.predict(query)
        logger.debug("gpt_response=%s" % response)
        self.last_txt.append(response)
        return response
