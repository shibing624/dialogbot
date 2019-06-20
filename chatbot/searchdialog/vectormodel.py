# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
import time

from chatbot.searchdialog.pos import POS_WEIGHT
from .vectorreader import load_corpus_model
from ..util.tokenizer import segment, segment_pos


class VectorModel:
    def __init__(self, corpus_file,
                 word2id,
                 similarity_type='word',
                 pos_weight=None):
        time_s = time.time()
        self.pos_weight = pos_weight or POS_WEIGHT
        self.similarity_type = similarity_type
        self.contexts, self.responses = self.load_corpus_file(corpus_file, word2id)

        self.corpus, self.embedding = load_corpus_model(train_data_path=corpus_file)
        print("Time to build vector model by %s : %2.f seconds." % (corpus_file, time.time() - time_s))

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
        if self.similarity_type == 'word':
            # simple word name overlapping coefficient
            score = len(set(l1) & set(l2)) / len(set(l2))
        elif self.similarity_type == 'word_pos':
            # word and pos overlapping coefficient
            sim_weight = 0
            for word, pos in set(l1):
                sim_weight += self.pos_weight.get(pos, 1) if word in l2 else 0
            total_weight = sum(self.pos_weight.get(pos, 1) for _, pos in set(l1))
            score = sim_weight / total_weight if total_weight > 0 else 0
        elif self.similarity_type == 'vector':
            # word vector and pos weight
            sim_weight = 0
            total_weight = 0
            for word, pos in l1:
                if word not in self.embedding.index2word:
                    continue
                cur_weight = self.pos_weight.get(pos, 1)
                max_word_sim = max([self.embedding.similarity(word_l2, word) for word_l2 in l2])
                sim_weight += cur_weight * max_word_sim
                total_weight += cur_weight
            score = sim_weight / total_weight if total_weight > 0 else 0
        else:
            print('error, not define similarity type')
            raise ValueError('error, not define similarity type')
        return score

    def similarity(self, query, size=10):
        """
        get the most similar question with input sentence
        :param query:
        :param size:
        :return:
        """
        for k, v in self.corpus.items():
            if self.similarity_type == 'word':
                question = v['question_segment']
                query_vec = segment(query)
            elif self.similarity_type == 'word_pos':
                question = v['question_segment_pos']
                query_vec = segment_pos(query)
            elif self.similarity_type == 'vector':
                question = v.get('question_vector', '')
                query_vec = [q_word for q_word in v['question_segment'] if q_word in self.embedding.index2word]
            else:
                print('error, not define similarity type')
                raise ValueError('error, not define similarity type')
            # add score to dict
            v['similarity_score'] = self.score(query_vec, question)
        # max_score = max(self.corpus.values(), key=lambda k: k['similarity_score'])
        corpus_scores_sort = sorted(list(self.corpus.values()), key=lambda k: k['similarity_score'], reverse=True)
        scores_sort = []
        for i in corpus_scores_sort:
            scores_sort.append((i['answer'], i['similarity_score']))

        return scores_sort[:size]

    def get_docs(self, sim_items):
        docs = [self.contexts[id_] for id_, score in sim_items]
        answers = [self.responses[id_] for id_, score in sim_items]
        return docs, answers
