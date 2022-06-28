# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: 
import time
from loguru import logger
from dialogbot.reader.data_helper import load_corpus_file


class OneHotModel:
    def __init__(self, corpus_file, word2id):
        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2id)
        logger.debug("Time to build onehot model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

    def score(self, l1, l2):
        """
        get similarity score by text vector and pos vector
        :param l1: input sentence list
        :param l2: sentence list which to be compared
        :return:
        """
        score = 0
        if not l1 or not l2:
            return score
        down = l1 if len(l1) > len(l2) else l2
        # simple word name overlapping coefficient
        score = len(set(l1) & set(l2)) / len(set(down))
        return score

    def similarity(self, query, size=10):
        """
        get the most similar question with input sentence
        :param query: segment tokens (list)
        :param size:
        :return:
        """
        scores = []
        for question in self.contexts:
            score = self.score(query, question)
            scores.append(score)
        scores_sort = sorted(list(enumerate(scores)), key=lambda item: item[1], reverse=True)
        return scores_sort[:size]

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
