# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: 
from collections import deque
from domain.reader import load_corpus_model
import config
from domain.util import segment, segment_pos, get_logger
from domain.similarity import word_pos_similarity

logger = get_logger(__name__)


class QA:
    def __init__(self, last_txt_len=10, train_file_path=None, train_model_path=None, emb_model_path=None):
        self.last_txt = deque([], last_txt_len)
        self.data, self.vec_model = load_corpus_model(train_model_path, emb_model_path, train_file_path)

    def max_similarity_score(self, sentence, similarity_score_threshold=0.4, similarity_type='word'):
        """
        get the most similar question with input sentence
        :param sentence:
        :param similarity_score_threshold:
        :param similarity_type:
        :return:
        """
        self.last_txt.append(sentence)
        if similarity_type not in ['word', 'word_pos', 'vector']:
            return 'error, similarity type not exists.'
        # not embedding, use word_pos
        embedding = self.vec_model
        if similarity_type == 'vector' and not embedding:
            similarity_type = 'word_pos'
        for k, v in self.data.items():
            question = v['question_vector'] if similarity_type == 'vector' else v['question_segment']
            sentence_vector = segment(sentence) if similarity_type == 'word' else segment_pos(sentence)
            # add score to dict
            v['similarity_score'] = word_pos_similarity(sentence_vector, question, similarity_type=similarity_type,
                                                        embedding=embedding)
        max_similarity = max(self.data.values(), key=lambda k: k['similarity_score'])
        logger.info('max question similarity score=' + format(max_similarity['similarity_score'], '.0%'))

        if max_similarity['similarity_score'] < similarity_score_threshold:
            return 'sorry, not understand your question.'
        return max_similarity['answer']

    def answer(self, sentence, similarity_type='vector'):
        """
        answer the question
        :param sentence:
        :param similarity_type:
        :return:
        """
        if not sentence:
            return ''

        if similarity_type == 'all':
            for type in ['word', 'word_pos', 'vector']:
                out = 'similarity_type:' + type + ' => ' + self.max_similarity_score(sentence, similarity_type=type)
                logger.info(out)
            return ''
        else:
            out = self.max_similarity_score(sentence, similarity_type=similarity_type)
        return out


if __name__ == '__main__':
    qa = QA(train_file_path=config.train_file_path, emb_model_path=config.emb_model_path,
            train_model_path=config.train_model_path)
    q = '如何去坚持减肥？'
    # q = 'nihao如何 '
    answer = qa.answer(q, 'all')
    print('question:', q, '\tanswer', answer)
