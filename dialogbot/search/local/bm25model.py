# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief:
from loguru import logger
from dialogbot.reader.data_helper import load_corpus_file
from dialogbot.search.local.rank_bm25 import BM25Okapi


class BM25Model:
    def __init__(self, corpus_file, word2id):
        self.contexts, self.responses = load_corpus_file(corpus_file, word2id)
        self.bm25_inst = BM25Okapi(self.contexts)
        logger.debug(f"Build bm25 model from {corpus_file}, contexts size: {len(self.contexts)}, responses size: {len(self.responses)}")

    def similarity(self, query, size=10):
        scores = self.bm25_inst.get_scores(query)

        scores_sort = sorted(list(enumerate(scores)),
                             key=lambda item: item[1], reverse=True)
        return scores_sort[:size]

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
