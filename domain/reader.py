# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import os
from util import segment_pos, segment


def load_corpus(data_path=None, emb_model_path=None):
    corpus = dict()
    vec_model = None
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts and len(parts) > 1:
                q, a = parts[0], parts[1]
                corpus[q] = {'answer': a, 'question_segment': segment(q), 'answer_segment': segment(a),
                             'question_segment_pos': segment_pos(q), 'answer_segment_pos': segment_pos(a)}
    if emb_model_path:
        corpus, vec_model = _add_embedding(emb_model_path, corpus)
    return corpus, vec_model


def _add_embedding(emb_model_path, corpus):
    from gensim.models import Word2Vec
    if not os.path.exists(emb_model_path):
        return corpus, None
    vec_model = Word2Vec.load(emb_model_path)
    for question in corpus.keys():
        q_vec = []
        for q_word in question:
            q_vec.append({i for i in q_word if i in vec_model.index2word})
        if q_vec:
            corpus[question]['question_vector'] = q_vec
    return corpus, vec_model


if __name__ == '__main__':
    load_corpus()
