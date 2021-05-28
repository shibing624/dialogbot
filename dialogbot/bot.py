# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: chat bot main process.
"""

from dialogbot import config
from dialogbot.searchdialog.bot import SearchBot
from dialogbot.seq2seqdialog.bot import Seq2SeqBot
from dialogbot.utils.text_util import ch_count


class Bot:
    def __init__(self,
                 vocab_path=config.vocab_path,
                 search_model=config.search_model,
                 question_answer_path=config.question_answer_path,
                 context_response_path=config.context_response_path,
                 seq2seq_model_path=config.seq2seq_model_path,
                 context=None):
        self.context = context if context else []
        self.search_bot = SearchBot(question_answer_path, context_response_path,
                                    vocab_path=vocab_path,
                                    search_model=search_model)
        self.seq2seq_bot = Seq2SeqBot(vocab_path, seq2seq_model_path)

    def set_context(self, v):
        if isinstance(v, list):
            self.context = v
        elif isinstance(v, str):
            self.context = [v]
        else:
            self.context = []

    def answer(self, msg, use_task=True):
        """
        Dialog strategy: use sub-task to handle dialog firstly,
        if failed, use retrieval or generational func to handle it.
        :param msg: str, input query
        :param use_task: bool,
        :return: response
        """
        task_response = ''
        if use_task:
            task_response = ''

        # Search response.
        if len(self.context) >= 3 and ch_count(msg) <= 4:
            # user_msgs = self.context[::2][-3:]
            # msg = "<s>".join(user_msgs)
            # mode = "cr"
            mode = "qa"
        else:
            mode = "qa"
        search_response, sim_score = self.search_bot.answer(msg, mode=mode)

        # Seq2seq response.
        seq2seq_response = self.seq2seq_bot.answer(msg)

        response = {"task_response": task_response,
                    "search_response": search_response,
                    "seq2seq_response": seq2seq_response,
                    }
        return response
