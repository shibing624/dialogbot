# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import os
import time

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from dialogbot.reader.data_helper import load_corpus_file
from dialogbot.utils.logger import logger


class VectorModel:
    def __init__(self, corpus_file,
                 word2id,
                 doc2vec_model_path='d2v.pkl'):
        time_s = time.time()
        self.contexts, self.responses = load_corpus_file(corpus_file, word2id, size=500)
        self.doc_model = self.load_doc2vec_model(self.contexts, doc2vec_model_path)
        logger.info("Time to build vector model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

    @staticmethod
    def load_doc2vec_model(texts, model_path):
        if os.path.exists(model_path):
            model = Word2Vec.load(model_path)
        else:
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
            model = Doc2Vec(documents, vector_size=5, window=3, min_count=1, workers=4, size=100, alpha=0.025, iter=40)
            model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
            # model = Word2Vec(sg=1, sentences=texts, size=256, window=5, min_count=1, iter=40)
            # model.save(model_path)
        return model

    def score(self, l1, l2):
        """
        get similarity score by wmd
        :param l1: input sentence list
        :param l2: sentence list which to be compared
        :return:
        """
        score = 0
        if not l1 or not l2:
            return score
        score = self.doc_model.wmdistance(l1, l2)
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
