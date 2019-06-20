# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import time
from codecs import open

from gensim import corpora, models, similarities
from chatbot.util.logger import get_logger

logger = get_logger(__name__)


class TfidfModel:
    def __init__(self, corpus_file, word2id, DEBUG_MODE=True):
        time_s = time.time()
        size = 1000 if DEBUG_MODE else 10000000
        self.contexts, self.responses = self.load_corpus_file(corpus_file, word2id, size)

        self._train_model()
        self.corpus_mm = self.tfidf_model[self.corpus]
        self.index = similarities.MatrixSimilarity(self.corpus_mm)
        logger.debug("Time to build tfidf model by %s: %2.f seconds." % (corpus_file, time.time() - time_s))

    @staticmethod
    def load_corpus_file(corpus_file, word2id, size):
        with open(corpus_file, "r", encoding="utf-8") as rfd:
            data = [s.strip().split("\t") for s in rfd.readlines()[:size]]
            contexts = [[w for w in s.split() if w in word2id] for s, _ in data]
            responses = [s.replace(" ", "") for _, s in data]
            return contexts, responses

    def _train_model(self, min_freq=1):
        # Create tfidf model.
        self.dct = corpora.Dictionary(self.contexts)
        # Filter low frequency words from dictionary.
        low_freq_ids = [id_ for id_, freq in
                        self.dct.dfs.items() if freq <= min_freq]
        self.dct.filter_tokens(low_freq_ids)
        self.dct.compactify()
        # Build tfidf-model.
        self.corpus = [self.dct.doc2bow(s) for s in self.contexts]
        self.tfidf_model = models.TfidfModel(self.corpus)

    def _text2vec(self, text):
        bow = self.dct.doc2bow(text)
        return self.tfidf_model[bow]

    def similarity(self, query, size=10):
        vec = self._text2vec(query)
        sims = self.index[vec]
        sim_sort = sorted(list(enumerate(sims)),
                          key=lambda item: item[1], reverse=True)
        return sim_sort[:size]

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
