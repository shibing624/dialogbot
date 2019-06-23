# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import time

from dialogbot.utils.logger import get_logger

logger = get_logger(__name__)


class OneHotModel:
    def __init__(self, corpus_file, word2id):
        time_s = time.time()
        self.contexts, self.responses = self.load_corpus_file(corpus_file, word2id)
        logger.debug("Time to build onehot model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

    @staticmethod
    def load_corpus_file(corpus_file, word2id, size=0):
        with open(corpus_file, "r", encoding="utf-8") as r:
            all_data = r.readlines()
            all_data = all_data[:size] if size > 0 else all_data
            data = [s.strip().split("\t") for s in all_data]
            contexts = [[w for w in s.split() if w in word2id] for s, _ in data]
            responses = [s.replace(" ", "") for _, s in data]
            return contexts, responses

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
