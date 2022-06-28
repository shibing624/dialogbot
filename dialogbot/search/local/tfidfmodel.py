# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import time
from loguru import logger
from gensim import corpora, models, similarities

from dialogbot.reader.data_helper import load_corpus_file


class TfidfModel:
    def __init__(self, corpus_file, word2id):
        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2id, size=50000)

        self._train_model()
        self.corpus_mm = self.tfidf_model[self.corpus]
        self.index = similarities.MatrixSimilarity(self.corpus_mm)
        logger.debug("Time to build tfidf model by %s: %2.f seconds." % (corpus_file, time.time() - time_s))

    def _train_model(self, min_freq=1):
        # Create tfidf model.
        self.dct = corpora.Dictionary(self.contexts)
        # Filter low frequency words from dictionary.
        low_freq_ids = [id_ for id_, freq in
                        self.dct.dfs.items() if freq <= min_freq]
        self.dct.filter_tokens(low_freq_ids)
        self.dct.compactify()
        # Build tfidf model.
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
