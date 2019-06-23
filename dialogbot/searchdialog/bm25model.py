# -*- coding: utf-8 -*-


import time
from codecs import open

from dialogbot.searchdialog.bm25 import BM25
from dialogbot.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Model:
    def __init__(self, corpus_file, word2id, DEBUG_MODE=True):
        time_s = time.time()
        size = 500000 if DEBUG_MODE else 10000000
        self.contexts, self.responses = self.load_corpus_file(corpus_file, word2id, size)
        self.bm25_inst = BM25(self.contexts)
        logger.debug("Time to build bm25 model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

    @staticmethod
    def load_corpus_file(corpus_file, word2id, size):
        with open(corpus_file, "r", encoding="utf-8") as rfd:
            data = [s.strip().split("\t") for s in rfd.readlines()[:size]]
            contexts = [[w for w in s.split() if w in word2id] for s, _ in data]
            responses = [s.replace(" ", "") for _, s in data]
            return contexts, responses

    def similarity(self, query, size=10):
        return self.bm25_inst.similarity(query, size)

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
