# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from collections import deque
from dialogbot.gpt.interact import Inference


class GPTBot:
    def __init__(
            self,
            model_name_or_path="shibing624/gpt2-dialogbot-base-chinese",
            max_history_len=3, max_len=128,
            repetition_penalty=1.0, temperature=1.0,
            topk=8, topp=0.0, last_txt_len=100
    ):
        self.last_txt = deque([], last_txt_len)
        self.model = Inference(
            model_name_or_path,
            max_history_len=max_history_len,
            max_len=max_len,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            topk=topk,
            topp=topp
        )

    def answer(self, query):
        response = ''
        if not self.model:
            return response
        self.last_txt.append(query)
        response = self.model.predict(query)
        self.last_txt.append(response)
        return response
