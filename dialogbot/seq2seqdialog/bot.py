# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
from collections import deque

from dialogbot.reader.data_helper import load_dataset
from dialogbot.seq2seqdialog.infer import get_infer_model, predict
from dialogbot.util.logger import get_logger

logger = get_logger(__name__)


class Seq2SeqBot:
    def __init__(self,
                 vocab_path=None,
                 dialog_mode="single",
                 last_txt_len=100):
        self.last_txt = deque([], last_txt_len)
        self.dialog_mode = dialog_mode
        self.word2id, self.id2word = load_dataset(vocab_path)
        self.model = get_infer_model(dialog_mode=dialog_mode)

    def answer(self, query):
        self.last_txt.append(query)
        logger.debug('-' * 20)
        logger.debug("init_query=%s" % query)
        response = predict(self.model, query, beam_size=1)
        logger.debug("seq2seq_response=%s" % response)
        self.last_txt.append(response)
        return response
